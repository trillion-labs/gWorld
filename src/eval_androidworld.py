#!/usr/bin/env python3
"""
World Model Evaluation Script - Android World

Evaluates MWM models by predicting S_t+1 from S_t and A_t using
state transitions extracted from deduplicated Android World pickle files.

Training format:
- Prompt includes reasoning instruction before HTML generation
- 9 requirements (includes reasoning requirement)
- Action WITHOUT backticks (direct text)
- Output format: # Next State Reasoning: ... # HTML: <html_code>
- Image processing: max_pixels=4233600, min_pixels=3136

Automatically skips already-completed evaluations, allowing you to add new models
without re-running existing evaluations.

Usage:
    python eval_androidworld.py
"""

import gc
import gzip
import io
import json
import os
import pickle
import re
import shutil
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from threading import Thread

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from playwright.sync_api import sync_playwright


def get_scale_factor_for_size(ref_width: int, ref_height: int) -> float:
    """
    Get appropriate scale factor for given image dimensions based on common device sizes.

    Returns: scale_factor
    """
    size_to_scale = {
        (1080, 2400): 3.0,   # Smartphone
        (1440, 3120): 4.0,   # Smartphone
        (1440, 3040): 4.0,   # Smartphone
        (720, 1280): 2.0,    # Smartphone
        (1344, 2992): 3.0,   # Smartphone (large screen/foldable)
        (1440, 2960): 4.0,   # Smartphone
        (1080, 2280): 3.0,   # Smartphone
        (1080, 2160): 3.0,   # Smartphone
        (2560, 1600): 2.0,   # Tablet
        (1600, 2560): 2.0,   # Tablet
        (2208, 1840): 2.0,   # Tablet
        (1840, 2208): 2.0,   # Tablet
    }

    if (ref_width, ref_height) in size_to_scale:
        return size_to_scale[(ref_width, ref_height)]

    is_portrait = ref_height > ref_width

    if is_portrait:
        for scale in [4.0, 3.0, 2.5, 2.0, 1.5]:
            logical_w = int(ref_width / scale)
            logical_h = int(ref_height / scale)
            if 300 <= logical_w <= 500 and 500 <= logical_h <= 1200:
                return scale
    else:
        for scale in [2.0, 1.5, 3.0]:
            logical_w = int(ref_width / scale)
            logical_h = int(ref_height / scale)
            if 600 <= logical_w <= 1200 and 400 <= logical_h <= 800:
                return scale

    return 2.0


def estimate_logical_viewport_size(ref_width: int, ref_height: int, scale_factor: float) -> tuple[int, int]:
    """Calculate logical viewport size from reference image using specified scale factor."""
    if ref_width <= 0 or ref_height <= 0:
        return (480, 800)

    logical_w = int(ref_width / scale_factor)
    logical_h = int(ref_height / scale_factor)

    return (logical_w, logical_h)


# Number of workers for parallel input preparation
INPUT_PREP_WORKERS = 64


# World Model User Prompt Template
WORLD_MODEL_USER_PROMPT_CONTENT = """You are an expert mobile UI World Model that can accurately predict the next state given an action.
Given a screenshot of a mobile interface and an action, you must generate clean, responsive HTML code that represents the state of the interface AFTER the action is performed.
First generate reasoning about what the next state should look like based on the action.
Afterwards, generate the HTML code representing the next state that logically follows the action.
You will render this HTML in a mobile viewport to see how similar it looks and acts like the mobile screenshot.

Requirements:
1. Provide reasoning about what the next state should look like based on the action
2. Generate complete, valid HTML5 code
3. Choose between using inline CSS and utility classes from Bootstrap, Tailwind CSS, or MUI for styling, depending on which option generates the closest code to the screenshot.
4. Use mobile-first design principles matching screenshot dimensions.
5. For images, use inline SVG placeholders with explicit width and height attributes that match the approximate dimensions from the screenshot. Matching the approximate color is also good.
6. Use modern web standards and best practices
7. Return ONLY the HTML code, no explanations or markdown formatting
8. The generated HTML should render properly in a mobile viewport.
9. Generated HTML should look like the screen that logically follows the current screen and the action.

Action:
{action}

Output format:
# Next State Reasoning: <your reasoning about what the next state should look like>
# HTML: <valid_html_code>

Generate the next state reasoning and the next state in html:"""


def cleanup_vllm_processes():
    """Aggressively clean up any lingering vLLM processes and GPU memory."""
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    try:
        current_pid = os.getpid()
        result = subprocess.run(
            ["pgrep", "-f", "vllm.entrypoints|vllm.worker|from vllm"],
            capture_output=True, text=True
        )
        if result.stdout.strip():
            for pid_str in result.stdout.strip().split('\n'):
                try:
                    pid = int(pid_str)
                    if pid != current_pid:
                        os.kill(pid, signal.SIGKILL)
                        print(f"    Killed orphan vLLM process: {pid}")
                except (ValueError, ProcessLookupError, PermissionError):
                    pass
    except Exception:
        pass

    try:
        subprocess.run(["pkill", "-9", "-f", "ray::"], capture_output=True)
    except Exception:
        pass

    time.sleep(3)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cleanup_gpu_memory_hard():
    """More aggressive GPU memory cleanup."""
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)

    time.sleep(2)


def convert_coords_to_qwen3_format(x: float, y: float, img_width: int, img_height: int) -> list[int]:
    """Convert pixel coordinates to Qwen3 format (0-1000)."""
    x_qwen3 = round(1000 * x / img_width)
    y_qwen3 = round(1000 * y / img_height)
    x_qwen3 = max(0, min(1000, x_qwen3))
    y_qwen3 = max(0, min(1000, y_qwen3))
    return [x_qwen3, y_qwen3]


def decompress_gz_files(tree_dir: str) -> int:
    """Find all .pkl.gz files, decompress them, and remove the .gz files."""
    decompressed_count = 0

    for gz_file in Path(tree_dir).glob("*.pkl.gz"):
        pkl_file = gz_file.with_suffix("")

        print(f"Decompressing: {gz_file.name} -> {pkl_file.name}")

        with gzip.open(gz_file, "rb") as f_in:
            with open(pkl_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        gz_file.unlink()
        decompressed_count += 1

    return decompressed_count


def extract_transitions(pkl_path: Path) -> list[dict]:
    """
    Extract state transitions from a pickle file.

    Returns list of transitions, each containing:
        - task_name: Name of the task
        - transition_idx: Index of transition within task
        - s_t: Current state screenshot (PIL Image or bytes)
        - a_t: Action taken (string)
        - s_t1: Next state screenshot (PIL Image or bytes)
        - goal: Task goal description
    """
    transitions = []

    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, list) and len(data) > 0:
            item = data[0]
        elif isinstance(data, dict):
            item = data
        else:
            return transitions

        episode_data = item.get("episode_data")
        if episode_data is None or (isinstance(episode_data, float) and np.isnan(episode_data)):
            return transitions

        if not isinstance(episode_data, dict):
            return transitions

        screenshots = episode_data.get("raw_screenshot", [])
        actions = episode_data.get("action_output_json", [])
        ui_elements_list = episode_data.get("before_ui_elements", [])
        goal = item.get("goal", "")

        if not screenshots or not actions:
            return transitions

        num_transitions = min(len(screenshots) - 1, len(actions))

        for i in range(num_transitions):
            ui_elements = ui_elements_list[i] if i < len(ui_elements_list) else []

            current_screenshot = screenshots[i]
            if isinstance(current_screenshot, Image.Image):
                img_width, img_height = current_screenshot.size
            elif isinstance(current_screenshot, np.ndarray):
                img_height, img_width = current_screenshot.shape[:2]
            elif isinstance(current_screenshot, bytes):
                img = Image.open(io.BytesIO(current_screenshot))
                img_width, img_height = img.size
            else:
                img_width, img_height = 1080, 1920

            action_data = actions[i]
            if isinstance(action_data, str):
                try:
                    action_data = json.loads(action_data)
                except json.JSONDecodeError:
                    action_str = action_data
                else:
                    action_str = extract_action_string(action_data, ui_elements, img_width, img_height)
            elif isinstance(action_data, dict):
                action_str = extract_action_string(action_data, ui_elements, img_width, img_height)
            else:
                action_str = extract_action_string(action_data, ui_elements, img_width, img_height)

            transition = {
                "task_name": pkl_path.stem,
                "transition_idx": i,
                "s_t": screenshots[i],
                "a_t": action_str,
                "s_t1": screenshots[i + 1],
                "goal": goal,
            }
            transitions.append(transition)

    except Exception as e:
        print(f"Error extracting transitions from {pkl_path.name}: {e}")

    return transitions


def extract_action_string(action_data, ui_elements=None, img_width=1080, img_height=1920) -> str:
    """Convert action to JSON string matching training data format."""
    if hasattr(action_data, '__dict__'):
        action_dict = action_data.__dict__
        action_type = action_dict.get('action_type', '')

        result = {"action": action_type}

        has_coords = False
        pixel_x, pixel_y = None, None

        if action_dict.get('x') is not None and action_dict.get('y') is not None:
            pixel_x = action_dict['x']
            pixel_y = action_dict['y']
            has_coords = True

        if not has_coords and action_dict.get('index') is not None and ui_elements:
            index = action_dict['index']
            if index < len(ui_elements):
                elem = ui_elements[index]
                if hasattr(elem, 'bbox_pixels') and elem.bbox_pixels is not None:
                    bbox = elem.bbox_pixels
                    pixel_x = (bbox.x_min + bbox.x_max) // 2
                    pixel_y = (bbox.y_min + bbox.y_max) // 2
                    has_coords = True

        if has_coords and pixel_x is not None and pixel_y is not None:
            qwen3_coords = convert_coords_to_qwen3_format(pixel_x, pixel_y, img_width, img_height)
            result["coordinates"] = qwen3_coords

        if action_dict.get('text') is not None:
            result["text"] = action_dict['text']
        if action_dict.get('direction') is not None:
            result["direction"] = action_dict['direction']
        if action_dict.get('app_name') is not None:
            result["app_name"] = action_dict['app_name']

        return json.dumps(result)

    if isinstance(action_data, dict):
        return json.dumps(action_data)

    if isinstance(action_data, str):
        return action_data

    return str(action_data)


def prepare_vllm_input(s_t, a_t: str, processor: AutoProcessor):
    """
    Prepare input for vLLM inference.

    Args:
        s_t: Current state image (bytes, numpy array, or PIL Image)
        a_t: Action string
        processor: HuggingFace processor for the model (handles chat template)
    """
    if isinstance(s_t, bytes):
        image = Image.open(io.BytesIO(s_t))
    elif isinstance(s_t, np.ndarray):
        image = Image.fromarray(s_t)
        if image.mode != 'RGB':
            image = image.convert('RGB')
    elif isinstance(s_t, Image.Image):
        image = s_t
        if image.mode != 'RGB':
            image = image.convert('RGB')
    else:
        raise ValueError(f"Unknown image type: {type(s_t)}")

    user_content = WORLD_MODEL_USER_PROMPT_CONTENT.format(action=a_t)

    # Build messages in standard VLM format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_content},
            ],
        }
    ]

    # Use the processor's chat template to format the prompt
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return {
        "prompt": prompt,
        "multi_modal_data": {"image": image},
    }


def prepare_single_input(args):
    """
    Prepare a single input for vLLM inference (for parallel execution).

    Args:
        args: Tuple of (transition, processor, html_dir, rendered_dir, reasoning_dir, shared_gt_dir, shared_input_dir)

    Returns:
        Tuple of (vllm_input, metadata) or (None, metadata) on error
    """
    transition, processor, html_dir, rendered_dir, reasoning_dir, shared_gt_dir, shared_input_dir = args
    task_name = transition["task_name"]
    t_idx = transition["transition_idx"]
    base_name = f"{task_name}_t{t_idx}"

    try:
        vllm_input = prepare_vllm_input(
            s_t=transition["s_t"],
            a_t=transition["a_t"],
            processor=processor,
        )

        metadata = {
            'transition': transition,
            'task_name': task_name,
            't_idx': t_idx,
            'base_name': base_name,
            'html_path': html_dir / f"{base_name}_pred.html",
            'rendered_path': rendered_dir / f"{base_name}_rendered.png",
            'reasoning_path': reasoning_dir / f"{base_name}_reasoning.txt",
            'gt_path': shared_gt_dir / f"{base_name}_gt.png",
            'input_path': shared_input_dir / f"{base_name}_input.png",
        }

        return (vllm_input, metadata)
    except Exception as e:
        return (None, {'task_name': task_name, 't_idx': t_idx, 'error': str(e), 'transition': transition})


def prepare_batch_parallel(transitions, processor, html_dir, rendered_dir, reasoning_dir, shared_gt_dir, shared_input_dir, max_workers=INPUT_PREP_WORKERS):
    """
    Prepare a batch of inputs in parallel.

    Returns:
        Tuple of (batch_inputs, batch_metadata, errors)
    """
    batch_inputs = []
    batch_metadata = []
    errors = []

    args_list = [
        (transition, processor, html_dir, rendered_dir, reasoning_dir, shared_gt_dir, shared_input_dir)
        for transition in transitions
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(prepare_single_input, args_list))

    for vllm_input, metadata in results:
        if vllm_input is not None:
            batch_inputs.append(vllm_input)
            batch_metadata.append(metadata)
        else:
            errors.append(metadata)

    return batch_inputs, batch_metadata, errors


class AsyncBatchPrefetcher:
    """
    Prefetches the next batch of inputs while GPU processes the current batch.

    This overlaps CPU-bound input preparation with GPU-bound inference.
    """

    def __init__(self, transitions_to_eval, batch_size, processor, html_dir, rendered_dir, reasoning_dir, shared_gt_dir, shared_input_dir):
        self.transitions = transitions_to_eval
        self.batch_size = batch_size
        self.processor = processor
        self.html_dir = html_dir
        self.rendered_dir = rendered_dir
        self.reasoning_dir = reasoning_dir
        self.shared_gt_dir = shared_gt_dir
        self.shared_input_dir = shared_input_dir

        self.num_batches = (len(transitions_to_eval) + batch_size - 1) // batch_size
        self.prefetch_queue = Queue(maxsize=2)  # Buffer up to 2 batches
        self.prefetch_thread = None
        self._stop = False

    def start(self):
        """Start the prefetch thread."""
        self._stop = False
        self.prefetch_thread = Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()

    def stop(self):
        """Stop the prefetch thread."""
        self._stop = True
        if self.prefetch_thread:
            self.prefetch_thread.join(timeout=5)

    def _prefetch_worker(self):
        """Background worker that prepares batches."""
        for batch_idx in range(self.num_batches):
            if self._stop:
                break

            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(self.transitions))
            batch_transitions = self.transitions[start_idx:end_idx]

            # Prepare batch in parallel
            batch_inputs, batch_metadata, errors = prepare_batch_parallel(
                batch_transitions,
                self.processor,
                self.html_dir,
                self.rendered_dir,
                self.reasoning_dir,
                self.shared_gt_dir,
                self.shared_input_dir,
            )

            # Put in queue (blocks if queue is full, which is what we want)
            self.prefetch_queue.put({
                'batch_idx': batch_idx,
                'batch_inputs': batch_inputs,
                'batch_metadata': batch_metadata,
                'errors': errors,
                'start_idx': start_idx,
                'end_idx': end_idx,
            })

        # Signal end of batches
        self.prefetch_queue.put(None)

    def get_next_batch(self):
        """Get the next prepared batch (blocks if not ready yet)."""
        return self.prefetch_queue.get()


def clean_html_response(raw_output: str) -> tuple[str, str]:
    """
    Clean HTML response by extracting reasoning and HTML.

    Output format:
    # Next State Reasoning: <reasoning>
    # HTML: <html_code>

    Returns:
        tuple of (reasoning, html)
    """
    reasoning = ""
    html = ""

    # Try to extract reasoning
    reasoning_match = re.search(
        r'# Next State Reasoning:\s*(.+?)(?=# HTML:|$)',
        raw_output,
        re.DOTALL
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    # Try to extract HTML after "# HTML:" marker
    html_match = re.search(r'# HTML:\s*(.+)', raw_output, re.DOTALL)
    if html_match:
        html = html_match.group(1).strip()
    else:
        # Fallback: if no marker found, try markdown code blocks
        if "```html" in raw_output:
            html = raw_output.split("```html")[1].split("```")[0]
        elif "```" in raw_output:
            html = raw_output.split("```")[1].split("```")[0]
        else:
            # Last resort: use everything after any HTML marker
            html = raw_output

    # Clean up any remaining markdown formatting
    if "```html" in html:
        html = html.split("```html")[1].split("```")[0]
    elif "```" in html:
        parts = html.split("```")
        if len(parts) >= 2:
            html = parts[1].split("```")[0] if parts[1] else html

    return reasoning.strip(), html.strip()


def save_ground_truth_image(image_data, output_path: Path):
    """Save ground truth image to file."""
    if isinstance(image_data, Image.Image):
        image_data.save(output_path)
    elif isinstance(image_data, bytes):
        with open(output_path, "wb") as f:
            f.write(image_data)
    elif isinstance(image_data, np.ndarray):
        img = Image.fromarray(image_data)
        img.save(output_path)
    else:
        raise ValueError(f"Unknown image type: {type(image_data)}")


def render_html_to_image(html_path: Path, output_path: Path, viewport_width: int = None, viewport_height: int = None, reference_image: Path = None, scale_factor: float = None) -> bool:
    """Render HTML file to a screenshot image using Playwright."""
    try:
        if viewport_width is None or viewport_height is None:
            if reference_image and reference_image.exists():
                try:
                    ref_img = Image.open(reference_image)
                    ref_width, ref_height = ref_img.size

                    if scale_factor is not None:
                        viewport_width, viewport_height = estimate_logical_viewport_size(ref_width, ref_height, scale_factor)
                    else:
                        max_dimension = 2048
                        if ref_width > max_dimension or ref_height > max_dimension:
                            scale = max_dimension / max(ref_width, ref_height)
                            viewport_width = int(ref_width * scale)
                            viewport_height = int(ref_height * scale)
                        else:
                            viewport_width = ref_width
                            viewport_height = ref_height
                except Exception as e:
                    print(f"    Warning: Could not read reference image {reference_image}: {e}")
                    viewport_width = viewport_width or 480
                    viewport_height = viewport_height or 800
            else:
                viewport_width = viewport_width or 480
                viewport_height = viewport_height or 800

        dpr = scale_factor if scale_factor is not None else 1

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={'width': viewport_width, 'height': viewport_height},
                device_scale_factor=dpr
            )
            page = context.new_page()
            page.goto(f"file://{html_path.absolute()}")
            page.wait_for_load_state('networkidle')
            page.screenshot(path=str(output_path), full_page=False)
            browser.close()
        return True
    except Exception as e:
        print(f"    Error rendering HTML: {e}")
        return False


def get_model_slug(model_name: str) -> str:
    """Convert model name to filesystem-safe slug."""
    return model_name.replace("/", "_").replace(":", "_")


def main():
    # Clean up any leftover processes
    print("Cleaning up any leftover vLLM processes...")
    cleanup_vllm_processes()

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    # Model configurations
    MODEL_BASE_PATH = "/home/work/.shared/sungjun/Qwen3-VL/qwen-vl-finetune/outputs/MWM-v12-32b"

    MODELS = [
        # =====================================================================
        # MWM-32b: Full fine-tuned 32B model with reasoning
        # =====================================================================
        {
            "name": f"{MODEL_BASE_PATH}/checkpoint-18790",
            "display_name": "MWM-32B-ckpt18790",
            "base_model": "Qwen/Qwen3-VL-32B-Instruct",
            "tensor_parallel_size": 8,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 19384,
        },
        {
            "name": f"{MODEL_BASE_PATH}/checkpoint-15032",
            "display_name": "MWM-32B-ckpt15032",
            "base_model": "Qwen/Qwen3-VL-32B-Instruct",
            "tensor_parallel_size": 8,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 19384,
        },
        {
            "name": f"{MODEL_BASE_PATH}/checkpoint-11274",
            "display_name": "MWM-32B-ckpt11274",
            "base_model": "Qwen/Qwen3-VL-32B-Instruct",
            "tensor_parallel_size": 8,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 19384,
        },
        {
            "name": f"{MODEL_BASE_PATH}/checkpoint-7516",
            "display_name": "MWM-32B-ckpt7516",
            "base_model": "Qwen/Qwen3-VL-32B-Instruct",
            "tensor_parallel_size": 8,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 19384,
        },
        {
            "name": f"{MODEL_BASE_PATH}/checkpoint-3758",
            "display_name": "MWM-32B-ckpt3758",
            "base_model": "Qwen/Qwen3-VL-32B-Instruct",
            "tensor_parallel_size": 8,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 19384,
        },
        # # =====================================================================
        # # Baselines (uncomment as needed)
        # # =====================================================================
        # {
        #     "name": "Qwen/Qwen3-VL-32B-Instruct",
        #     "display_name": "Qwen3-VL-32B-baseline",
        #     "base_model": "Qwen/Qwen3-VL-32B-Instruct",
        #     "is_lora": False,
        #     "tensor_parallel_size": 8,
        #     "gpu_memory_utilization": 0.9,
        #     "max_model_len": 19384,
        # },
        # {
        #     "name": "Qwen/Qwen3-VL-235B-A22B-Instruct",
        #     "display_name": "Qwen3-VL-235B-A22B-baseline",
        #     "base_model": "Qwen/Qwen3-VL-235B-A22B-Instruct",
        #     "tensor_parallel_size": 8,
        #     "gpu_memory_utilization": 0.9,
        #     "max_model_len": 19384,
        # },
    ]

    # GPU configuration
    CUDA_VISIBLE_DEVICES = "0,1,2,3,4,5,6,7"

    # Input/Output directories
    TREE_DIR = "/home/work/.shared/segyu/mwm/eval_wm/Tree_dedup"
    OUTPUT_BASE_DIR = "eval_outputs"

    # Shared ground truth and inputs (reused across all pickle evaluations)
    SHARED_GT_DIR = Path("pickle_ground_truth")
    SHARED_INPUT_DIR = Path("pickle_inputs")

    # Evaluation limits (set to None for all)
    MAX_TASKS = None
    MAX_TRANSITIONS_PER_TASK = None

    # Generation parameters
    MAX_TOKENS = 15000
    TEMPERATURE = 0
    BATCH_SIZE = 1028

    # Parallel workers settings
    IMAGE_COPY_WORKERS = 48  # Parallel image saving workers
    RENDER_WORKERS = 48  # Parallel HTML rendering workers

    # vLLM engine settings
    USE_V1_ENGINE = True

    # vLLM optimization settings - balanced for stability
    VLLM_ENABLE_CHUNKED_PREFILL = True  # Better memory scheduling
    VLLM_MAX_NUM_BATCHED_TOKENS = 16384  # Reduced from 65536 - safer for VLM with images

    # Image processing settings - match training config
    MM_PROCESSOR_KWARGS = {
        "max_pixels": 4233600,
        "min_pixels": 3136,
    }

    # =========================================================================
    # END CONFIGURATION
    # =========================================================================

    if not USE_V1_ENGINE:
        os.environ["VLLM_USE_V1"] = "0"
        print("Using vLLM v0 engine")

    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

    print("=" * 70)
    print("WORLD MODEL EVALUATION (Android World)")
    print("=" * 70)
    print(f"\nGPU: {CUDA_VISIBLE_DEVICES}")
    print(f"Models to evaluate: {len(MODELS)}")
    for i, model_cfg in enumerate(MODELS, 1):
        display = model_cfg.get('display_name', model_cfg['name'])
        print(f"  {i}. {display}")

    # Check tree directory
    if not os.path.exists(TREE_DIR):
        print(f"Error: Directory '{TREE_DIR}' not found")
        return

    # Decompress any .pkl.gz files
    print("\n[1] Checking for compressed files...")
    decompressed = decompress_gz_files(TREE_DIR)
    if decompressed > 0:
        print(f"    Decompressed {decompressed} files")

    # Load all transitions
    print("\n[2] Loading state transitions from pickle files...")
    pkl_files = sorted(Path(TREE_DIR).glob("*.pkl"))

    all_transitions = []
    tasks_processed = 0

    for pkl_file in pkl_files:
        if MAX_TASKS and tasks_processed >= MAX_TASKS:
            break

        transitions = extract_transitions(pkl_file)

        if transitions:
            if MAX_TRANSITIONS_PER_TASK:
                transitions = transitions[:MAX_TRANSITIONS_PER_TASK]
            all_transitions.extend(transitions)
            tasks_processed += 1
            print(f"    {pkl_file.stem}: {len(transitions)} transitions")

    print(f"\n    Total transitions: {len(all_transitions)}")

    if not all_transitions:
        print("No transitions found. Exiting.")
        return

    # Use shared ground truth and inputs directories
    shared_gt_dir = SHARED_GT_DIR
    shared_input_dir = SHARED_INPUT_DIR

    # Check if shared directories exist, create and populate if needed
    shared_gt_dir.mkdir(parents=True, exist_ok=True)
    shared_input_dir.mkdir(parents=True, exist_ok=True)

    print("\n[3] Saving ground truth and input images...")

    def save_single_image(args):
        """Save a single image (for parallel execution)."""
        image_data, output_path = args
        if output_path.exists():
            return "skipped"
        try:
            save_ground_truth_image(image_data, output_path)
            return "saved"
        except Exception as e:
            return f"error: {e}"

    # Build list of image save tasks
    image_tasks = []
    for transition in all_transitions:
        task_name = transition["task_name"]
        t_idx = transition["transition_idx"]
        base_name = f"{task_name}_t{t_idx}"

        gt_path = shared_gt_dir / f"{base_name}_gt.png"
        input_path = shared_input_dir / f"{base_name}_input.png"

        image_tasks.append((transition["s_t1"], gt_path))
        image_tasks.append((transition["s_t"], input_path))

    stats = {"skipped": 0, "saved": 0, "error": 0}

    with ThreadPoolExecutor(max_workers=IMAGE_COPY_WORKERS) as executor:
        futures = {executor.submit(save_single_image, task): task for task in image_tasks}
        with tqdm(total=len(image_tasks), desc="    Processing images") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result == "skipped":
                    stats["skipped"] += 1
                elif result == "saved":
                    stats["saved"] += 1
                else:
                    stats["error"] += 1
                pbar.update(1)

    print(f"    Image processing complete:")
    print(f"      - Skipped (already exist): {stats['skipped']}")
    print(f"      - Saved: {stats['saved']}")
    if stats['error'] > 0:
        print(f"      - Errors: {stats['error']}")

    # Evaluate each model
    for model_idx, model_cfg in enumerate(MODELS, 1):
        model_name = model_cfg["name"]
        model_slug = get_model_slug(model_cfg.get("display_name", model_name))

        print("\n" + "=" * 70)
        print(f"EVALUATING MODEL {model_idx}/{len(MODELS)}: {model_cfg.get('display_name', model_name)}")
        print("=" * 70)

        # Create output directories
        output_dir = Path(OUTPUT_BASE_DIR) / model_slug
        output_dir.mkdir(parents=True, exist_ok=True)

        html_dir = output_dir / "predictions"
        rendered_dir = output_dir / "rendered"
        reasoning_dir = output_dir / "reasoning"
        html_dir.mkdir(parents=True, exist_ok=True)
        rendered_dir.mkdir(parents=True, exist_ok=True)
        reasoning_dir.mkdir(parents=True, exist_ok=True)

        # Check which transitions need evaluation
        transitions_to_eval = []
        skipped_count = 0

        for transition in all_transitions:
            task_name = transition["task_name"]
            t_idx = transition["transition_idx"]
            base_name = f"{task_name}_t{t_idx}"

            html_path = html_dir / f"{base_name}_pred.html"
            rendered_path = rendered_dir / f"{base_name}_rendered.png"

            if html_path.exists() and rendered_path.exists():
                skipped_count += 1
            else:
                transitions_to_eval.append(transition)

        if skipped_count > 0:
            print(f"\n    Skipping {skipped_count} already completed transitions")

        if not transitions_to_eval:
            print(f"    All transitions completed. Skipping model.")
            continue

        print(f"    Transitions to evaluate: {len(transitions_to_eval)}")

        # Initialize model
        base_model = model_cfg.get("base_model", "Qwen/Qwen3-VL-32B-Instruct")
        is_baseline = model_name == base_model

        print(f"\n    Loading model...")
        print(f"    Base model: {base_model}")
        print(f"    Type: {'Baseline' if is_baseline else 'Full fine-tuned'}")

        # vLLM kwargs
        llm_kwargs = {
            "model": model_name,
            "tokenizer": base_model,
            "tensor_parallel_size": model_cfg["tensor_parallel_size"],
            "gpu_memory_utilization": model_cfg["gpu_memory_utilization"],
            "max_model_len": model_cfg["max_model_len"],
            "trust_remote_code": True,
            # Optimization settings for higher GPU utilization
            "enable_chunked_prefill": VLLM_ENABLE_CHUNKED_PREFILL,
            "max_num_batched_tokens": VLLM_MAX_NUM_BATCHED_TOKENS,
            "disable_log_stats": True,  # Reduce logging overhead
        }

        # Override custom architecture for fine-tuned Qwen models
        if "Qwen" in base_model and not is_baseline:
            llm_kwargs["hf_overrides"] = {"architectures": ["Qwen3VLForConditionalGeneration"]}

        # Add mm_processor_kwargs only for Qwen models (Llama 4 has different image processing)
        if "Qwen" in base_model:
            llm_kwargs["mm_processor_kwargs"] = MM_PROCESSOR_KWARGS

        # Add limit_mm_per_prompt if specified (needed for Llama 4)
        if "limit_mm_per_prompt" in model_cfg:
            llm_kwargs["limit_mm_per_prompt"] = model_cfg["limit_mm_per_prompt"]

        llm = LLM(**llm_kwargs)

        # Load processor for chat template (use base model)
        print(f"    Loading processor...")
        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)

        sampling_params = SamplingParams(
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            seed=42,
            top_p=1.0,
        )

        # Load existing results
        results = []
        results_path = output_dir / "eval_results.json"

        if results_path.exists():
            print(f"\n    Loading existing results...")
            try:
                with open(results_path, "r") as f:
                    results = json.load(f)
                print(f"    Found {len(results)} existing results")
            except Exception as e:
                print(f"    Warning: Could not load existing results: {e}")
                results = []

        # Process in batches with async prefetching
        num_batches = (len(transitions_to_eval) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\n    Processing {len(transitions_to_eval)} transitions in {num_batches} batches...")
        print(f"    Using async prefetching with {INPUT_PREP_WORKERS} parallel workers")
        print("    " + "-" * 66)

        # Start async prefetcher
        prefetcher = AsyncBatchPrefetcher(
            transitions_to_eval=transitions_to_eval,
            batch_size=BATCH_SIZE,
            processor=processor,
            html_dir=html_dir,
            rendered_dir=rendered_dir,
            reasoning_dir=reasoning_dir,
            shared_gt_dir=shared_gt_dir,
            shared_input_dir=shared_input_dir,
        )
        prefetcher.start()

        try:
            while True:
                # Get next prepared batch (blocks if not ready)
                batch_data = prefetcher.get_next_batch()
                if batch_data is None:
                    break  # No more batches

                batch_idx = batch_data['batch_idx']
                batch_inputs = batch_data['batch_inputs']
                batch_metadata = batch_data['batch_metadata']
                prep_errors = batch_data['errors']
                start_idx = batch_data['start_idx']
                end_idx = batch_data['end_idx']

                print(f"\n    Batch {batch_idx + 1}/{num_batches} (transitions {start_idx + 1}-{end_idx})")

                # Record any preparation errors
                for err in prep_errors:
                    results.append({
                        "task_name": err['task_name'],
                        "transition_idx": err['t_idx'],
                        "action": err['transition']["a_t"],
                        "goal": err['transition']["goal"],
                        "status": "error",
                        "error": f"Input preparation failed: {err['error']}",
                        "model": model_cfg.get('display_name', model_name),
                    })

                if not batch_inputs:
                    print(f"        No valid inputs in batch, skipping...")
                    continue

                try:
                    # Generate batch
                    print(f"        Generating {len(batch_inputs)} predictions...")
                    outputs = llm.generate(batch_inputs, sampling_params)

                    # Process outputs - save HTMLs and reasoning first (fast)
                    render_tasks = []  # Collect rendering tasks
                    for idx, (output, meta) in enumerate(zip(outputs, batch_metadata)):
                        transition = meta['transition']
                        task_name = meta['task_name']
                        t_idx = meta['t_idx']

                        try:
                            raw_output = output.outputs[0].text
                            reasoning, predicted_html = clean_html_response(raw_output)

                            output_obj = output.outputs[0]
                            finish_reason = output_obj.finish_reason if hasattr(output_obj, 'finish_reason') else "unknown"
                            num_output_tokens = len(output_obj.token_ids) if hasattr(output_obj, 'token_ids') else -1

                            if finish_reason == "length":
                                print(f"        [{task_name}_t{t_idx}] WARNING: Hit max token limit!")
                            if len(predicted_html) == 0:
                                print(f"        [{task_name}_t{t_idx}] WARNING: Empty HTML output!")

                            # Save HTML (fast)
                            with open(meta['html_path'], "w") as f:
                                f.write(predicted_html)

                            # Save reasoning (fast)
                            with open(meta['reasoning_path'], "w") as f:
                                f.write(reasoning)

                            # Queue for parallel rendering
                            render_tasks.append({
                                'html_path': meta['html_path'],
                                'rendered_path': meta['rendered_path'],
                                'task_name': task_name,
                                't_idx': t_idx,
                                'transition': transition,
                                'meta': meta,
                                'num_output_tokens': num_output_tokens,
                                'finish_reason': finish_reason,
                                'html_length': len(predicted_html),
                                'reasoning_length': len(reasoning),
                            })

                        except Exception as e:
                            print(f"        [{task_name}_t{t_idx}] Error: {e}")
                            results.append({
                                "task_name": task_name,
                                "transition_idx": t_idx,
                                "action": transition["a_t"],
                                "goal": transition["goal"],
                                "status": "error",
                                "error": str(e),
                                "model": model_cfg.get('display_name', model_name),
                            })

                    # Parallel rendering with ThreadPoolExecutor
                    def render_single(task):
                        try:
                            # Calculate scale factor from input image
                            input_path = task['meta']['input_path']
                            sf = None
                            if input_path.exists():
                                try:
                                    ref_img = Image.open(input_path)
                                    ref_w, ref_h = ref_img.size
                                    sf = get_scale_factor_for_size(ref_w, ref_h)
                                except Exception:
                                    pass
                            success = render_html_to_image(
                                task['html_path'],
                                task['rendered_path'],
                                reference_image=input_path,
                                scale_factor=sf
                            )
                            return {**task, 'render_success': success}
                        except Exception as e:
                            return {**task, 'render_success': False, 'render_error': str(e)}

                    print(f"        Rendering {len(render_tasks)} HTMLs in parallel...")
                    with ThreadPoolExecutor(max_workers=RENDER_WORKERS) as executor:
                        render_results = list(executor.map(render_single, render_tasks))

                    # Collect results
                    for task in render_results:
                        transition = task['transition']
                        task_name = task['task_name']
                        t_idx = task['t_idx']
                        meta = task['meta']
                        render_success = task['render_success']

                        results.append({
                            "task_name": task_name,
                            "transition_idx": t_idx,
                            "action": transition["a_t"],
                            "goal": transition["goal"],
                            "prediction_path": f"{model_slug}/predictions/{meta['html_path'].name}",
                            "rendered_path": f"{model_slug}/rendered/{meta['rendered_path'].name}" if render_success else None,
                            "reasoning_path": f"{model_slug}/reasoning/{meta['reasoning_path'].name}",
                            "ground_truth_path": f"pickle_ground_truth/{meta['gt_path'].name}",
                            "input_path": f"pickle_inputs/{meta['input_path'].name}",
                            "status": "success",
                            "render_status": "success" if render_success else "failed",
                            "model": model_cfg.get('display_name', model_name),
                            "output_tokens": task['num_output_tokens'],
                            "finish_reason": task['finish_reason'],
                            "html_length": task['html_length'],
                            "reasoning_length": task['reasoning_length'],
                        })

                    rendered_count = sum(1 for t in render_results if t['render_success'])
                    print(f"        Rendered {rendered_count}/{len(render_tasks)} successfully")

                    # Save after each batch
                    with open(results_path, "w") as f:
                        json.dump(results, f, indent=2)

                except Exception as e:
                    import traceback
                    print(f"\n        Batch error: {e}")
                    traceback.print_exc()

                    for meta in batch_metadata:
                        transition = meta['transition']
                        results.append({
                            "task_name": transition["task_name"],
                            "transition_idx": transition["transition_idx"],
                            "action": transition["a_t"],
                            "goal": transition["goal"],
                            "status": "error",
                            "error": f"Batch failed: {str(e)}",
                            "model": model_cfg.get('display_name', model_name),
                        })

                    with open(results_path, "w") as f:
                        json.dump(results, f, indent=2)

                    # Check if engine died - if so, break and let retry mechanism handle it
                    err_str = str(e).lower()
                    if 'enginedead' in err_str or 'enginecore' in err_str:
                        print("\n        FATAL: vLLM engine died. Breaking to allow retry...")
                        raise  # Re-raise to trigger script restart via retry mechanism

        finally:
            # Always stop the prefetcher
            prefetcher.stop()

        # Add skipped results
        existing_entries = {(r["task_name"], r["transition_idx"]) for r in results}

        for transition in all_transitions:
            task_name = transition["task_name"]
            t_idx = transition["transition_idx"]
            base_name = f"{task_name}_t{t_idx}"

            if (task_name, t_idx) in existing_entries:
                continue

            html_path = html_dir / f"{base_name}_pred.html"
            rendered_path = rendered_dir / f"{base_name}_rendered.png"
            reasoning_path = reasoning_dir / f"{base_name}_reasoning.txt"
            gt_path = shared_gt_dir / f"{base_name}_gt.png"
            input_path = shared_input_dir / f"{base_name}_input.png"

            if transition not in transitions_to_eval and html_path.exists() and rendered_path.exists():
                results.append({
                    "task_name": task_name,
                    "transition_idx": t_idx,
                    "action": transition["a_t"],
                    "goal": transition["goal"],
                    "prediction_path": f"{model_slug}/predictions/{html_path.name}",
                    "rendered_path": f"{model_slug}/rendered/{rendered_path.name}",
                    "reasoning_path": f"{model_slug}/reasoning/{reasoning_path.name}",
                    "ground_truth_path": f"pickle_ground_truth/{gt_path.name}",
                    "input_path": f"pickle_inputs/{input_path.name}",
                    "status": "success",
                    "render_status": "success",
                    "skipped": True,
                    "model": model_cfg.get('display_name', model_name),
                })

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Summary
        print("\n    " + "-" * 66)
        print(f"    MODEL COMPLETE: {model_cfg.get('display_name', model_name)}")
        print("    " + "-" * 66)

        successful = sum(1 for r in results if r.get("status") == "success")
        failed = sum(1 for r in results if r.get("status") == "error")
        rendered = sum(1 for r in results if r.get("render_status") == "success")
        skipped = sum(1 for r in results if r.get("skipped", False))

        print(f"\n    Total: {len(results)}")
        print(f"      - Successful: {successful}")
        print(f"      - Failed: {failed}")
        print(f"      - Rendered: {rendered}")
        if skipped > 0:
            print(f"      - Skipped: {skipped}")
        print(f"\n    Results: {results_path}")

        # Cleanup
        print("\n    Cleaning up GPU memory...")
        del llm
        cleanup_gpu_memory_hard()
        cleanup_vllm_processes()
        print("    Cleanup complete.")

    # Final summary
    print("\n" + "=" * 70)
    print("ALL MODELS EVALUATED")
    print("=" * 70)
    print(f"\nOutputs saved to: {OUTPUT_BASE_DIR}")


if __name__ == "__main__":
    main()
