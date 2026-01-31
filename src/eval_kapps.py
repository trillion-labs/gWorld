#!/usr/bin/env python3
"""
World Model Evaluation Script - Korean Apps (KApps)

Evaluates world model checkpoints on the KApps benchmark dataset.
The KApps benchmark is a filtered/downsampled version of the original benchmark,
containing 500 pre-selected transitions from Korean mobile applications.

Data format (filtered-500):
- JSON files organized by app (baemin, coupang, etc.)
- Each JSON contains exactly ONE transition (2 states: current and next)
- Filename format: {app}_{task}_{timestamp}_transition{idx}.json
- Each file has a trajectory list with 2 items: [current_state, next_state]
- States have screenshot_base64 (base64 encoded PNG)
- Actions are in the first trajectory item

Training format:
- Prompt includes reasoning instruction before HTML generation
- 9 requirements (includes reasoning requirement)
- Action WITHOUT backticks (direct text)
- Output format: # Next State Reasoning: ... # HTML: <html_code>
- Image processing: max_pixels=4233600, min_pixels=3136

Usage:
    python eval_kapps.py
"""

import base64
import io
import json
import os
import re
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from threading import Thread

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
INPUT_PREP_WORKERS = 32


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
    """
    Aggressively clean up any lingering vLLM processes and GPU memory.
    """
    import gc
    import torch

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
    import gc
    import torch

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


def decode_base64_image(base64_str: str) -> Image.Image:
    """Decode base64 encoded image string to PIL Image."""
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image


def extract_action_string(action: dict, img_width: int, img_height: int) -> str:
    """
    Convert action format to JSON string matching training data format.

    Action format:
        - action_type: TAP, SET_TEXT, SCROLL, LAUNCH_APP, TERMINATION, etc.
        - coordinates: {x, y} in pixels
        - text: text for SET_TEXT
        - direction: for SCROLL
        - app_name: for LAUNCH_APP

    Training format:
        {"action": "TAP", "coordinates": [x_qwen3, y_qwen3]}
        {"action": "SET_TEXT", "coordinates": [x_qwen3, y_qwen3], "text": "..."}
        {"action": "SCROLL", "direction": "up/down/left/right"}
        {"action": "LAUNCH_APP", "app_name": "..."}
    """
    action_type = action.get('action_type', '')

    result = {"action": action_type}

    # Handle coordinates
    coords = action.get('coordinates')
    if coords and coords.get('x') is not None and coords.get('y') is not None:
        pixel_x = coords['x']
        pixel_y = coords['y']
        qwen3_coords = convert_coords_to_qwen3_format(pixel_x, pixel_y, img_width, img_height)
        result["coordinates"] = qwen3_coords

    # Handle text
    if action.get('text') is not None:
        result["text"] = action['text']

    # Handle direction (for SCROLL)
    if action.get('direction') is not None:
        result["direction"] = action['direction']

    # Handle app_name (for LAUNCH_APP)
    if action.get('app_name') is not None:
        result["app_name"] = action['app_name']

    return json.dumps(result)


def load_kapps_samples(kapps_dir: Path) -> list[dict]:
    """
    Load all samples from the KApps benchmark directory.

    Format: Each JSON file contains exactly ONE transition (2 states).
    The filename format is: {app}_{task}_{timestamp}_transition{idx}.json

    Returns list of samples, each containing:
        - id: Unique identifier (filename stem)
        - app: App name
        - task_name: Task name
        - instruction: Task instruction
        - s_t: Current state image (PIL Image)
        - a_t: Action string (JSON format)
        - s_t1: Next state image (PIL Image)
        - transition_idx: Index extracted from filename
    """
    samples = []
    skipped_termination = 0
    errors = 0

    for app_dir in sorted(kapps_dir.iterdir()):
        if not app_dir.is_dir() or app_dir.name.startswith('.'):
            continue

        # Skip deprecated directory
        if app_dir.name == 'deprecated':
            continue

        for json_file in sorted(app_dir.glob('*.json')):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                app = data.get('app', '')
                task_name = data.get('task_name', '')
                instruction = data.get('instruction', '')
                trajectory = data.get('trajectory', [])

                # Each file should have exactly 2 trajectory items
                if len(trajectory) != 2:
                    print(f"    Warning: {json_file.name} has {len(trajectory)} trajectory items (expected 2)")
                    errors += 1
                    continue

                # Get screen size from metadata
                metadata = data.get('metadata', {})
                device_size = metadata.get('device_size', {})
                img_width = device_size.get('width', 1080)
                img_height = device_size.get('height', 2280)

                # Current state and next state
                current = trajectory[0]
                next_state = trajectory[1]

                action = current.get('action', {})
                action_type = action.get('action_type', '')

                # Skip TERMINATION actions (should not exist in filtered data, but just in case)
                if action_type == 'TERMINATION':
                    skipped_termination += 1
                    continue

                # Get current screenshot
                current_state = current.get('state', {})
                current_screenshot_b64 = current_state.get('screenshot_base64', '')

                # Get next screenshot
                next_state_data = next_state.get('state', {})
                next_screenshot_b64 = next_state_data.get('screenshot_base64', '')

                if not current_screenshot_b64 or not next_screenshot_b64:
                    print(f"    Warning: {json_file.name} missing screenshot data")
                    errors += 1
                    continue

                # Decode images
                try:
                    s_t = decode_base64_image(current_screenshot_b64)
                    s_t1 = decode_base64_image(next_screenshot_b64)
                except Exception as e:
                    print(f"    Warning: {json_file.name} image decode error: {e}")
                    errors += 1
                    continue

                # Convert action to training format
                action_str = extract_action_string(action, img_width, img_height)

                # Use filename stem as sample ID (includes transition info)
                sample_id = json_file.stem

                # Extract transition index from filename if present
                transition_idx = -1
                if '_transition' in sample_id:
                    try:
                        transition_idx = int(sample_id.split('_transition')[-1])
                    except ValueError:
                        pass

                sample = {
                    'id': sample_id,
                    'app': app,
                    'task_name': task_name,
                    'instruction': instruction,
                    's_t': s_t,
                    'a_t': action_str,
                    's_t1': s_t1,
                    'transition_idx': transition_idx,
                    'img_width': img_width,
                    'img_height': img_height,
                }
                samples.append(sample)

            except Exception as e:
                print(f"    Error loading {json_file.name}: {e}")
                errors += 1

    if skipped_termination > 0:
        print(f"    Skipped {skipped_termination} TERMINATION actions")
    if errors > 0:
        print(f"    Encountered {errors} errors")

    return samples


def prepare_vllm_input(s_t: Image.Image, a_t: str, processor: AutoProcessor):
    """
    Prepare input for vLLM inference.

    Args:
        s_t: Current state image (PIL Image)
        a_t: Action string
        processor: HuggingFace processor for the model (handles chat template)
    """
    # Format user prompt content using the training template
    user_content = WORLD_MODEL_USER_PROMPT_CONTENT.format(action=a_t)

    # Build messages in standard VLM format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": s_t},
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
        "multi_modal_data": {"image": s_t},
    }


def prepare_single_input(args):
    """
    Prepare a single input for vLLM inference (for parallel execution).
    """
    sample, processor, html_dir, rendered_dir, reasoning_dir, shared_gt_dir, shared_input_dir = args
    sample_id = sample['id']

    try:
        vllm_input = prepare_vllm_input(
            s_t=sample['s_t'],
            a_t=sample['a_t'],
            processor=processor,
        )

        metadata = {
            'sample': sample,
            'sample_id': sample_id,
            'html_path': html_dir / f"{sample_id}_pred.html",
            'rendered_path': rendered_dir / f"{sample_id}_rendered.png",
            'reasoning_path': reasoning_dir / f"{sample_id}_reasoning.txt",
            'gt_path': shared_gt_dir / f"{sample_id}_gt.png",
            'input_path': shared_input_dir / f"{sample_id}_input.png",
        }

        return (vllm_input, metadata)
    except Exception as e:
        return (None, {'sample_id': sample_id, 'error': str(e), 'sample': sample})


def prepare_batch_parallel(samples, processor, html_dir, rendered_dir, reasoning_dir, shared_gt_dir, shared_input_dir, max_workers=INPUT_PREP_WORKERS):
    """
    Prepare a batch of inputs in parallel.
    """
    batch_inputs = []
    batch_metadata = []
    errors = []

    args_list = [
        (sample, processor, html_dir, rendered_dir, reasoning_dir, shared_gt_dir, shared_input_dir)
        for sample in samples
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
    """

    def __init__(self, samples_to_eval, batch_size, processor, html_dir, rendered_dir, reasoning_dir, shared_gt_dir, shared_input_dir):
        self.samples = samples_to_eval
        self.batch_size = batch_size
        self.processor = processor
        self.html_dir = html_dir
        self.rendered_dir = rendered_dir
        self.reasoning_dir = reasoning_dir
        self.shared_gt_dir = shared_gt_dir
        self.shared_input_dir = shared_input_dir

        self.num_batches = (len(samples_to_eval) + batch_size - 1) // batch_size
        self.prefetch_queue = Queue(maxsize=2)
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
            end_idx = min(start_idx + self.batch_size, len(self.samples))
            batch_samples = self.samples[start_idx:end_idx]

            batch_inputs, batch_metadata, errors = prepare_batch_parallel(
                batch_samples,
                self.processor,
                self.html_dir,
                self.rendered_dir,
                self.reasoning_dir,
                self.shared_gt_dir,
                self.shared_input_dir,
            )

            self.prefetch_queue.put({
                'batch_idx': batch_idx,
                'batch_inputs': batch_inputs,
                'batch_metadata': batch_metadata,
                'errors': errors,
                'start_idx': start_idx,
                'end_idx': end_idx,
            })

        self.prefetch_queue.put(None)

    def get_next_batch(self):
        """Get the next prepared batch (blocks if not ready yet)."""
        return self.prefetch_queue.get()


def clean_html_response(raw_output: str) -> tuple[str, str]:
    """
    Clean HTML response by extracting reasoning and HTML.
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
            html = raw_output

    # Clean up any remaining markdown formatting
    if "```html" in html:
        html = html.split("```html")[1].split("```")[0]
    elif "```" in html:
        parts = html.split("```")
        if len(parts) >= 2:
            html = parts[1].split("```")[0] if parts[1] else html

    return reasoning.strip(), html.strip()


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

    # KApps benchmark directory (filtered 500 samples)
    KAPPS_DIR = Path("/home/work/.shared/segyu/mwm/eval_wm/sona-recordings-data-filtered-500")

    # Model configurations
    MODEL_32B_BASE_PATH = "/home/work/.shared/sungjun/Qwen3-VL/qwen-vl-finetune/outputs/MWM-v12-32b"
    MODEL_BASE_PATH = "/home/work/.shared/sungjun/Qwen3-VL/qwen-vl-finetune/outputs/MWM-v12-8b"

    MODELS = [
        # =====================================================================
        # MWM-8b: Full fine-tuned 8B model with reasoning
        # =====================================================================
        {
            "name": f"{MODEL_BASE_PATH}/checkpoint-18790",
            "display_name": "MWM-8B-ckpt18790",
            "base_model": "Qwen/Qwen3-VL-8B-Instruct",
            "tensor_parallel_size": 8,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 19384,
        },
        {
            "name": f"{MODEL_BASE_PATH}/checkpoint-15032",
            "display_name": "MWM-8B-ckpt15032",
            "base_model": "Qwen/Qwen3-VL-8B-Instruct",
            "tensor_parallel_size": 8,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 19384,
        },
        {
            "name": f"{MODEL_BASE_PATH}/checkpoint-11274",
            "display_name": "MWM-8B-ckpt11274",
            "base_model": "Qwen/Qwen3-VL-8B-Instruct",
            "tensor_parallel_size": 8,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 19384,
        },
        # # =====================================================================
        # # MWM-32b: Full fine-tuned 32B model with reasoning
        # # =====================================================================
        # {
        #     "name": f"{MODEL_32B_BASE_PATH}/checkpoint-18790",
        #     "display_name": "MWM-32B-ckpt18790",
        #     "base_model": "Qwen/Qwen3-VL-32B-Instruct",
        #     "tensor_parallel_size": 8,
        #     "gpu_memory_utilization": 0.9,
        #     "max_model_len": 19384,
        # },
        # # =====================================================================
        # # Baselines (uncomment as needed)
        # # =====================================================================
        # {
        #     "name": "Qwen/Qwen3-VL-8B-Instruct",
        #     "display_name": "Qwen3-VL-8B-baseline",
        #     "base_model": "Qwen/Qwen3-VL-8B-Instruct",
        #     "is_lora": False,
        #     "tensor_parallel_size": 8,
        #     "gpu_memory_utilization": 0.9,
        #     "max_model_len": 19384,
        # },
    ]

    # GPU and output config
    CUDA_VISIBLE_DEVICES = "0,1,2,3,4,5,6,7"
    OUTPUT_BASE_DIR = Path("eval_outputs_kapps")

    # Shared ground truth and inputs directories (reused across all evaluations)
    SHARED_GT_DIR = Path("kapps_ground_truth")
    SHARED_INPUT_DIR = Path("kapps_inputs")

    # Limits
    MAX_SAMPLES = None

    # Generation parameters
    MAX_TOKENS = 15000
    TEMPERATURE = 0
    BATCH_SIZE = 1028

    # Image processing settings - match training config
    MM_PROCESSOR_KWARGS = {
        "max_pixels": 4233600,
        "min_pixels": 3136,
    }

    # Parallel workers settings
    IMAGE_COPY_WORKERS = 48
    RENDER_WORKERS = 48

    # vLLM engine settings
    USE_V1_ENGINE = True

    # vLLM optimization settings
    VLLM_ENABLE_CHUNKED_PREFILL = True
    VLLM_MAX_NUM_BATCHED_TOKENS = 16384

    # =========================================================================
    # END CONFIGURATION
    # =========================================================================

    if not USE_V1_ENGINE:
        os.environ["VLLM_USE_V1"] = "0"
        print("Using vLLM v0 engine")

    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

    print("=" * 70)
    print("WORLD MODEL EVALUATION (Korean Apps - KApps)")
    print("=" * 70)
    print(f"\nGPU: {CUDA_VISIBLE_DEVICES}")
    print(f"KApps directory: {KAPPS_DIR}")
    print(f"Models to evaluate: {len(MODELS)}")
    for i, model_cfg in enumerate(MODELS, 1):
        display = model_cfg.get('display_name', model_cfg['name'])
        print(f"  {i}. {display}")

    # Check kapps directory exists
    if not KAPPS_DIR.exists():
        print(f"Error: KApps directory '{KAPPS_DIR}' not found")
        return

    # Load kapps samples
    print("\n[1] Loading KApps benchmark samples...")
    all_samples = load_kapps_samples(KAPPS_DIR)

    if MAX_SAMPLES:
        all_samples = all_samples[:MAX_SAMPLES]

    print(f"    Total samples: {len(all_samples)}")

    if not all_samples:
        print("No samples found. Exiting.")
        return

    # Create output and shared directories
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    shared_gt_dir = SHARED_GT_DIR
    shared_input_dir = SHARED_INPUT_DIR
    shared_gt_dir.mkdir(parents=True, exist_ok=True)
    shared_input_dir.mkdir(parents=True, exist_ok=True)

    # Save images in parallel
    print("\n[2] Saving ground truth and input images...")

    def save_single_image(args):
        sample_id, image, dst_path = args
        if dst_path.exists():
            return "skipped"
        try:
            image.save(dst_path, 'PNG')
            return "saved"
        except Exception as e:
            return f"error: {e}"

    image_tasks = []
    for sample in all_samples:
        sample_id = sample['id']
        gt_dst = shared_gt_dir / f"{sample_id}_gt.png"
        input_dst = shared_input_dir / f"{sample_id}_input.png"
        image_tasks.append((sample_id, sample['s_t1'], gt_dst))
        image_tasks.append((sample_id, sample['s_t'], input_dst))

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
        output_dir = OUTPUT_BASE_DIR / model_slug
        html_dir = output_dir / "predictions"
        rendered_dir = output_dir / "rendered"
        reasoning_dir = output_dir / "reasoning"
        html_dir.mkdir(parents=True, exist_ok=True)
        rendered_dir.mkdir(parents=True, exist_ok=True)
        reasoning_dir.mkdir(parents=True, exist_ok=True)

        # Check which samples need processing
        samples_to_eval = []
        skipped_count = 0

        for sample in all_samples:
            sample_id = sample['id']
            html_path = html_dir / f"{sample_id}_pred.html"
            rendered_path = rendered_dir / f"{sample_id}_rendered.png"

            if html_path.exists() and rendered_path.exists():
                skipped_count += 1
            else:
                samples_to_eval.append(sample)

        if skipped_count > 0:
            print(f"\n    Skipping {skipped_count} already completed samples")

        if not samples_to_eval:
            print("    All samples completed. Skipping model.")
            continue

        print(f"    Samples to evaluate: {len(samples_to_eval)}")

        # Initialize model
        base_model = model_cfg.get("base_model", "Qwen/Qwen3-VL-8B-Instruct")
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
            "enable_chunked_prefill": VLLM_ENABLE_CHUNKED_PREFILL,
            "max_num_batched_tokens": VLLM_MAX_NUM_BATCHED_TOKENS,
            "disable_log_stats": True,
        }

        # Override custom architecture for fine-tuned Qwen models
        if "Qwen" in base_model and not is_baseline:
            llm_kwargs["hf_overrides"] = {"architectures": ["Qwen3VLForConditionalGeneration"]}

        # Add mm_processor_kwargs only for Qwen models
        if "Qwen" in base_model:
            llm_kwargs["mm_processor_kwargs"] = MM_PROCESSOR_KWARGS

        # Add limit_mm_per_prompt if specified (needed for Llama 4)
        if "limit_mm_per_prompt" in model_cfg:
            llm_kwargs["limit_mm_per_prompt"] = model_cfg["limit_mm_per_prompt"]

        llm = LLM(**llm_kwargs)

        # Load processor for chat template
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
        num_batches = (len(samples_to_eval) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\n    Processing {len(samples_to_eval)} samples in {num_batches} batches...")
        print(f"    Using async prefetching with {INPUT_PREP_WORKERS} parallel workers")
        print("    " + "-" * 66)

        # Start async prefetcher
        prefetcher = AsyncBatchPrefetcher(
            samples_to_eval=samples_to_eval,
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
                # Get next prepared batch
                batch_data = prefetcher.get_next_batch()
                if batch_data is None:
                    break

                batch_idx = batch_data['batch_idx']
                batch_inputs = batch_data['batch_inputs']
                batch_metadata = batch_data['batch_metadata']
                prep_errors = batch_data['errors']
                start_idx = batch_data['start_idx']
                end_idx = batch_data['end_idx']

                print(f"\n    Batch {batch_idx + 1}/{num_batches} (samples {start_idx + 1}-{end_idx})")

                # Record any preparation errors
                for err in prep_errors:
                    results.append({
                        "sample_id": err['sample_id'],
                        "app": err['sample'].get('app', ''),
                        "task_name": err['sample'].get('task_name', ''),
                        "action": err['sample'].get('a_t', ''),
                        "status": "error",
                        "error": f"Input preparation failed: {err['error']}",
                        "model": model_cfg.get('display_name', model_name),
                    })

                if not batch_inputs:
                    print(f"        No valid inputs in batch, skipping...")
                    continue

                try:
                    # Generate
                    print(f"        Generating {len(batch_inputs)} predictions...")
                    outputs = llm.generate(batch_inputs, sampling_params)

                    # Process outputs
                    render_tasks = []
                    for idx, (output, meta) in enumerate(zip(outputs, batch_metadata)):
                        sample = meta['sample']
                        sample_id = meta['sample_id']

                        try:
                            raw_output = output.outputs[0].text
                            reasoning, html = clean_html_response(raw_output)

                            output_obj = output.outputs[0]
                            finish_reason = output_obj.finish_reason if hasattr(output_obj, 'finish_reason') else "unknown"
                            num_output_tokens = len(output_obj.token_ids) if hasattr(output_obj, 'token_ids') else -1

                            if finish_reason == "length":
                                print(f"        [{sample_id}] WARNING: Hit max token limit!")
                            if len(html) == 0:
                                print(f"        [{sample_id}] WARNING: Empty HTML output!")

                            # Save HTML
                            with open(meta['html_path'], "w") as f:
                                f.write(html)

                            # Save reasoning
                            with open(meta['reasoning_path'], "w") as f:
                                f.write(reasoning)

                            # Queue for parallel rendering
                            render_tasks.append({
                                'html_path': meta['html_path'],
                                'rendered_path': meta['rendered_path'],
                                'sample_id': sample_id,
                                'sample': sample,
                                'meta': meta,
                                'num_output_tokens': num_output_tokens,
                                'finish_reason': finish_reason,
                                'html_length': len(html),
                                'reasoning_length': len(reasoning),
                            })

                        except Exception as e:
                            print(f"        [{sample_id}] Error: {e}")
                            results.append({
                                "sample_id": sample_id,
                                "app": sample.get('app', ''),
                                "task_name": sample.get('task_name', ''),
                                "action": sample.get('a_t', ''),
                                "status": "error",
                                "error": str(e),
                                "model": model_cfg.get('display_name', model_name),
                            })

                    # Parallel rendering
                    def render_single(task):
                        try:
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
                        sample = task['sample']
                        sample_id = task['sample_id']
                        meta = task['meta']
                        render_success = task['render_success']

                        results.append({
                            "sample_id": sample_id,
                            "app": sample.get('app', ''),
                            "task_name": sample.get('task_name', ''),
                            "action": sample.get('a_t', ''),
                            "instruction": sample.get('instruction', ''),
                            "transition_idx": sample.get('transition_idx', -1),
                            "prediction_path": f"{model_slug}/predictions/{meta['html_path'].name}",
                            "rendered_path": f"{model_slug}/rendered/{meta['rendered_path'].name}" if render_success else None,
                            "reasoning_path": f"{model_slug}/reasoning/{meta['reasoning_path'].name}",
                            "ground_truth_path": f"kapps_ground_truth/{meta['gt_path'].name}",
                            "input_path": f"kapps_inputs/{meta['input_path'].name}",
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
                        sample = meta['sample']
                        results.append({
                            "sample_id": sample['id'],
                            "app": sample.get('app', ''),
                            "task_name": sample.get('task_name', ''),
                            "action": sample.get('a_t', ''),
                            "status": "error",
                            "error": f"Batch failed: {str(e)}",
                            "model": model_cfg.get('display_name', model_name),
                        })

                    with open(results_path, "w") as f:
                        json.dump(results, f, indent=2)

                    # Check if engine died
                    err_str = str(e).lower()
                    if 'enginedead' in err_str or 'enginecore' in err_str:
                        print("\n        FATAL: vLLM engine died. Breaking to allow retry...")
                        raise

        finally:
            prefetcher.stop()

        # Add skipped results
        existing_ids = {r["sample_id"] for r in results}
        for sample in all_samples:
            sample_id = sample['id']
            if sample_id in existing_ids:
                continue

            html_path = html_dir / f"{sample_id}_pred.html"
            rendered_path = rendered_dir / f"{sample_id}_rendered.png"

            if sample not in samples_to_eval and html_path.exists() and rendered_path.exists():
                results.append({
                    "sample_id": sample_id,
                    "app": sample.get('app', ''),
                    "task_name": sample.get('task_name', ''),
                    "action": sample.get('a_t', ''),
                    "instruction": sample.get('instruction', ''),
                    "transition_idx": sample.get('transition_idx', -1),
                    "prediction_path": f"{model_slug}/predictions/{html_path.name}",
                    "rendered_path": f"{model_slug}/rendered/{rendered_path.name}",
                    "ground_truth_path": f"kapps_ground_truth/{sample_id}_gt.png",
                    "input_path": f"kapps_inputs/{sample_id}_input.png",
                    "status": "success",
                    "render_status": "success",
                    "skipped": True,
                    "model": model_cfg.get('display_name', model_name),
                })

        # Final save
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
