#!/usr/bin/env python3
"""
World Model Evaluation Script - Test Splits

Evaluates gWorld/MWM models on the test split from the training data.

Training format:
- Prompt includes reasoning instruction before HTML generation
- 9 requirements (includes reasoning requirement)
- Action WITHOUT backticks (direct text)
- Output format: # Next State Reasoning: ... # HTML: <html_code>
- Image processing: max_pixels=4233600, min_pixels=3136

Automatically skips already-completed evaluations, allowing you to add new models
without re-running existing evaluations.

Usage:
    python eval_test_splits.py
"""

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


# Base directory for image files
IMAGE_BASE_DIR = Path("/home/work/.shared/data/mfm/images")


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


def resolve_image_path(relative_path: str, base_dir: Path = IMAGE_BASE_DIR) -> Path:
    """
    Convert JSONL relative path to actual file path.

    Path mappings:
    - android_control/images/episode_X/step_Y.jpg -> android_control/images/episode_X/step_Y.jpg
    - AMEX/filename.png -> AMEX/filename.png
    - guiodyssey/filename.png -> guiodyssey/filename.png
    - AitW/filename.jpg -> AitW/filename.jpg
    """
    return base_dir / relative_path


def parse_action_from_conversation(conversation_text: str) -> str:
    """
    Parse Action from the conversation text.

    The format has:
    ...
    Action:
    <action text>

    Output format:
    ...

    Returns:
        action string
    """
    action_match = re.search(
        r'Action:\n(.+?)\n\nOutput format:',
        conversation_text,
        re.DOTALL
    )
    if action_match:
        return action_match.group(1).strip()

    return ""


def load_test_samples(jsonl_path: Path) -> list[dict]:
    """
    Load test samples from JSONL file.

    Parses Action from the embedded conversation text.

    Returns list of samples, each containing:
        - id: Unique identifier
        - image_path: Resolved path to input image (S_t)
        - next_image_path: Resolved path to ground truth image (S_t+1)
        - action: Action string
        - task_type: Task type identifier
    """
    samples = []
    skipped = 0
    parse_errors = 0

    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)

            # Resolve image paths
            image_path = resolve_image_path(data['image'])
            next_image_path = resolve_image_path(data['next_image'])

            # Skip if images don't exist
            if not image_path.exists() or not next_image_path.exists():
                skipped += 1
                continue

            # Parse action from conversation
            conversation_text = data['conversations'][0]['value']
            action = parse_action_from_conversation(conversation_text)

            if not action:
                parse_errors += 1
                continue

            sample = {
                'id': data['id'],
                'episode_id': data.get('episode_id'),
                'step': data.get('step'),
                'image_path': image_path,
                'next_image_path': next_image_path,
                'action': action,
                'task_type': data.get('task_type', ''),
            }
            samples.append(sample)

    if skipped > 0:
        print(f"    Skipped {skipped} samples with missing images")
    if parse_errors > 0:
        print(f"    Skipped {parse_errors} samples with parse errors")

    return samples


def prepare_vllm_input(image_path: Path, action: str, processor: AutoProcessor):
    """
    Prepare input for vLLM inference.

    Args:
        image_path: Path to input image
        action: Action string
        processor: HuggingFace processor for the model (handles chat template)
    """
    # Load image
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Format user prompt content using the training template
    user_content = WORLD_MODEL_USER_PROMPT_CONTENT.format(action=action)

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
        args: Tuple of (sample, processor, html_dir, rendered_dir, reasoning_dir, shared_gt_dir, shared_input_dir)

    Returns:
        Tuple of (vllm_input, metadata) or (None, metadata) on error
    """
    sample, processor, html_dir, rendered_dir, reasoning_dir, shared_gt_dir, shared_input_dir = args
    sample_id = sample['id']

    try:
        vllm_input = prepare_vllm_input(
            image_path=sample['image_path'],
            action=sample['action'],
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

    Returns:
        Tuple of (batch_inputs, batch_metadata, errors)
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

    This overlaps CPU-bound input preparation with GPU-bound inference.
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
            end_idx = min(start_idx + self.batch_size, len(self.samples))
            batch_samples = self.samples[start_idx:end_idx]

            # Prepare batch in parallel
            batch_inputs, batch_metadata, errors = prepare_batch_parallel(
                batch_samples,
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

    # Test data
    # TEST_JSONL_PATH = Path("/home/work/.shared/sungjun/MWM/outputs_openrouter/training_v12_test_2000.jsonl") # This was our local path
    TEST_JSONL_PATH = Path("[Insert the path to MWMBench-TestSplits]")

    # Model configurations
    MODELS = [
        # =====================================================================
        # gWorld-8B
        # =====================================================================
        {
            "name": "trillionlabs/gWorld-8B",
            "display_name": "gWorld-8B",
            "base_model": "trillionlabs/gWorld-8B",
            "tensor_parallel_size": 8,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 19384,
        },
        # =====================================================================
        # gWorld-32B
        # =====================================================================
        {
            "name": "trillionlabs/gWorld-32B",
            "display_name": "gWorld-32B",
            "base_model": "trillionlabs/gWorld-32B",
            "tensor_parallel_size": 8,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 19384,
        },
    ]

    # GPU and output config
    CUDA_VISIBLE_DEVICES = "0,1,2,3,4,5,6,7"
    OUTPUT_BASE_DIR = Path("eval_outputs_test")

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
    RENDER_WORKERS = 48  # Parallel HTML rendering workers

    # vLLM engine settings
    USE_V1_ENGINE = True

    # vLLM optimization settings - balanced for stability
    VLLM_ENABLE_CHUNKED_PREFILL = True  # Better memory scheduling
    VLLM_MAX_NUM_BATCHED_TOKENS = 16384  # Reduced from 65536 - safer for VLM with images

    # =========================================================================
    # END CONFIGURATION
    # =========================================================================

    if not USE_V1_ENGINE:
        os.environ["VLLM_USE_V1"] = "0"
        print("Using vLLM v0 engine")

    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

    print("=" * 70)
    print("WORLD MODEL EVALUATION (Test Splits)")
    print("=" * 70)
    print(f"\nGPU: {CUDA_VISIBLE_DEVICES}")
    print(f"Test file: {TEST_JSONL_PATH}")
    print(f"Models to evaluate: {len(MODELS)}")
    for i, model_cfg in enumerate(MODELS, 1):
        display = model_cfg.get('display_name', model_cfg['name'])
        print(f"  {i}. {display}")

    # Check test file exists
    if not TEST_JSONL_PATH.exists():
        print(f"Error: Test file '{TEST_JSONL_PATH}' not found")
        return

    # Load test samples
    print("\n[1] Loading test samples from JSONL...")
    all_samples = load_test_samples(TEST_JSONL_PATH)

    if MAX_SAMPLES:
        all_samples = all_samples[:MAX_SAMPLES]

    print(f"    Total samples: {len(all_samples)}")

    if not all_samples:
        print("No samples found. Exiting.")
        return

    # Create shared directories
    shared_gt_dir = OUTPUT_BASE_DIR / "ground_truth"
    shared_input_dir = OUTPUT_BASE_DIR / "inputs"
    shared_gt_dir.mkdir(parents=True, exist_ok=True)
    shared_input_dir.mkdir(parents=True, exist_ok=True)

    # Copy images in parallel
    print("\n[2] Copying ground truth and input images...")

    def process_single_image(args):
        sample_id, src_path, dst_path = args
        if dst_path.exists():
            return "skipped"
        try:
            img = Image.open(src_path)
            img.save(dst_path, 'PNG')
            return "converted"
        except Exception as e:
            return f"error: {e}"

    image_tasks = []
    for sample in all_samples:
        sample_id = sample['id']
        gt_dst = shared_gt_dir / f"{sample_id}_gt.png"
        input_dst = shared_input_dir / f"{sample_id}_input.png"
        image_tasks.append((sample_id, sample['next_image_path'], gt_dst))
        image_tasks.append((sample_id, sample['image_path'], input_dst))

    stats = {"skipped": 0, "converted": 0, "error": 0}

    with ThreadPoolExecutor(max_workers=IMAGE_COPY_WORKERS) as executor:
        futures = {executor.submit(process_single_image, task): task for task in image_tasks}
        with tqdm(total=len(image_tasks), desc="    Processing images") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result == "skipped":
                    stats["skipped"] += 1
                elif result == "converted":
                    stats["converted"] += 1
                else:
                    stats["error"] += 1
                pbar.update(1)

    print(f"    Image processing complete:")
    print(f"      - Skipped (already exist): {stats['skipped']}")
    print(f"      - Converted from source: {stats['converted']}")
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
            # Optimization settings for higher GPU utilization
            "enable_chunked_prefill": VLLM_ENABLE_CHUNKED_PREFILL,
            "max_num_batched_tokens": VLLM_MAX_NUM_BATCHED_TOKENS,
            "disable_log_stats": True,  # Reduce logging overhead
        }

        # Override custom architecture for fine-tuned Qwen models
        if ("Qwen" in base_model or "gWorld" in model_name) and not is_baseline:
            # Only apply if the model actually needs this specific class mapping
            if "Qwen3" in base_model:
                llm_kwargs["hf_overrides"] = {"architectures": ["Qwen3VLForConditionalGeneration"]}

        # Add mm_processor_kwargs only for Qwen/gWorld models
        if "Qwen" in base_model or "gWorld" in model_name:
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

                print(f"\n    Batch {batch_idx + 1}/{num_batches} (samples {start_idx + 1}-{end_idx})")

                # Record any preparation errors
                for err in prep_errors:
                    results.append({
                        "sample_id": err['sample_id'],
                        "action": err['sample']['action'],
                        "task_type": err['sample']['task_type'],
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

                    # Process outputs - save HTMLs and reasoning first (fast)
                    render_tasks = []  # Collect rendering tasks
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

                            # Save HTML (fast)
                            with open(meta['html_path'], "w") as f:
                                f.write(html)

                            # Save reasoning (fast)
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
                                "action": sample['action'],
                                "task_type": sample['task_type'],
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
                        sample = task['sample']
                        sample_id = task['sample_id']
                        meta = task['meta']
                        render_success = task['render_success']

                        results.append({
                            "sample_id": sample_id,
                            "action": sample['action'],
                            "task_type": sample['task_type'],
                            "prediction_path": f"{model_slug}/predictions/{meta['html_path'].name}",
                            "rendered_path": f"{model_slug}/rendered/{meta['rendered_path'].name}" if render_success else None,
                            "reasoning_path": f"{model_slug}/reasoning/{meta['reasoning_path'].name}",
                            "ground_truth_path": f"ground_truth/{meta['gt_path'].name}",
                            "input_path": f"inputs/{meta['input_path'].name}",
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
                            "action": sample['action'],
                            "task_type": sample['task_type'],
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
                    "action": sample['action'],
                    "task_type": sample['task_type'],
                    "prediction_path": f"{model_slug}/predictions/{html_path.name}",
                    "rendered_path": f"{model_slug}/rendered/{rendered_path.name}",
                    "ground_truth_path": f"ground_truth/{sample_id}_gt.png",
                    "input_path": f"inputs/{sample_id}_input.png",
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