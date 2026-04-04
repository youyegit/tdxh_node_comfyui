import importlib.util
import json
import os
import sys
from pathlib import Path


from .kimi_nodes import _is_empty_image_input


PROVIDERS_DIR = Path(__file__).resolve().parent
API_NODES_DIR = PROVIDERS_DIR.parent
PLUGIN_ROOT_DIR = API_NODES_DIR.parent
COMFY_ROOT_DIR = PLUGIN_ROOT_DIR.parent.parent
QWENVL_GGUF_PATH = COMFY_ROOT_DIR / "custom_nodes" / "ComfyUI-QwenVL" / "AILab_QwenVL_GGUF.py"

_QWENVL_MODULE = None
QWENVL_GGUF_MODELS_JSON = COMFY_ROOT_DIR / "custom_nodes" / "ComfyUI-QwenVL" / "gguf_models.json"
CANONICAL_NODE_NAME = "TdxhLocalQwenVLDynamicVisionChat"


def _load_qwenvl_module():
    global _QWENVL_MODULE

    if _QWENVL_MODULE is not None:
        return _QWENVL_MODULE

    if not QWENVL_GGUF_PATH.is_file():
        raise FileNotFoundError(
            "ComfyUI-QwenVL is not installed. Expected file: "
            f"{QWENVL_GGUF_PATH}"
        )

    qwenvl_dir = str(QWENVL_GGUF_PATH.parent)
    if qwenvl_dir not in sys.path:
        sys.path.insert(0, qwenvl_dir)

    spec = importlib.util.spec_from_file_location("tdxh_local_qwenvl_gguf", str(QWENVL_GGUF_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load QwenVL GGUF module from {QWENVL_GGUF_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _QWENVL_MODULE = module
    return module


def _available_model_names():
    try:
        module = _load_qwenvl_module()
        if not QWENVL_GGUF_MODELS_JSON.is_file():
            raise FileNotFoundError(f"Missing gguf_models.json: {QWENVL_GGUF_MODELS_JSON}")

        data = json.loads(QWENVL_GGUF_MODELS_JSON.read_text(encoding="utf-8")) or {}
        base_dir = module._resolve_base_dir(data.get("base_dir") or "llm/GGUF")
        repos = data.get("qwenVL_model") or data.get("vl_repos") or data.get("repos") or {}
        model_keys = []

        for _, repo in sorted(repos.items()):
            if not isinstance(repo, dict):
                continue

            author = repo.get("author") or repo.get("publisher") or ""
            repo_name = repo.get("repo_name") or repo.get("repo_name_override") or ""
            mmproj_file = repo.get("mmproj_file")
            model_files = repo.get("model_files") or []
            if not mmproj_file or not model_files:
                continue

            author_dir = module._safe_dirname(str(author))
            repo_dir = module._safe_dirname(str(repo_name))
            target_dir = Path(base_dir) / author_dir / repo_dir
            mmproj_path = target_dir / Path(mmproj_file).name
            if not mmproj_path.is_file():
                continue

            for model_file in model_files:
                model_name = Path(model_file).name
                model_path = target_dir / model_name
                if model_path.is_file():
                    model_keys.append(model_name)

        model_keys = sorted(set(model_keys))

        if model_keys:
            return model_keys
    except Exception:
        pass
    return ["Qwen3VL-8B-Instruct-Q4_K_M.gguf"]


def _provider_name_for_model(model_name):
    stem = Path(model_name).stem
    return f"Local{stem}"


def get_local_qwenvl_provider_specs():
    return [(_provider_name_for_model(model_name), model_name) for model_name in _available_model_names()]


def get_local_qwenvl_provider_choices():
    return [provider_name for provider_name, _ in get_local_qwenvl_provider_specs()]


def get_local_qwenvl_model_name(provider_name):
    for candidate_provider_name, model_name in get_local_qwenvl_provider_specs():
        if candidate_provider_name == provider_name:
            return model_name
    return ""


def _should_retry_after_load_failure(message):
    lowered = str(message or "").lower()
    return "failed to load model from file" in lowered or "failed to initialize model" in lowered


def _run_backend_process(backend, payload, node_label):
    try:
        return backend.process(**payload)
    except Exception as exc:
        if not _should_retry_after_load_failure(exc):
            raise

        print(f"[{node_label}] WARN: {exc}. Clearing local QwenVL backend and retrying once.")
        clear_fn = getattr(backend, "clear", None)
        if callable(clear_fn):
            try:
                clear_fn()
            except Exception:
                pass

        retry_payload = dict(payload)
        retry_payload["keep_model_loaded"] = False
        return backend.process(**retry_payload)


def _is_empty_media_input(media):
    if media is None:
        return True
    if isinstance(media, (list, tuple)):
        if not media:
            return True
        return all(_is_empty_media_input(item) for item in media)
    return _is_empty_image_input(media)


def _collect_effective_images(inputcount, image_1, **kwargs):
    images = [image_1]
    for idx in range(2, int(inputcount) + 1):
        images.append(kwargs.get(f"image_{idx}"))

    effective_images = []
    for image in images:
        if _is_empty_image_input(image):
            continue
        effective_images.append(image)

    return effective_images


def _combine_prompt(system_prompt, prompt):
    system_prompt = str(system_prompt or "").strip()
    prompt = str(prompt or "").strip()

    if not prompt:
        raise ValueError(f"{CANONICAL_NODE_NAME}: prompt is empty.")

    if system_prompt:
        return f"{system_prompt}\n\n{prompt}"
    return prompt


class TdxhLocalQwenVLDynamicVisionChat:
    DESCRIPTION = (
        "Local QwenVL GGUF multimodal chat node wrapped for tdxh_node_comfyui. "
        "Uses the installed ComfyUI-QwenVL plugin as the backend. "
        "Supports a dynamic number of image inputs with 'Update inputs' and also accepts one fixed optional video input. "
        "Empty image slots are ignored, including a missing image_1. "
        "Image inputs and the fixed video input can be used together, and no-media requests fall back to text mode."
    )

    def __init__(self):
        self._backend_node = None
        self._text_backend = _TdxhLocalQwenVLTextBackend()

    @classmethod
    def INPUT_TYPES(cls):
        model_names = _available_model_names()
        default_model = "Qwen3VL-8B-Instruct-Q4_K_M.gguf" if "Qwen3VL-8B-Instruct-Q4_K_M.gguf" in model_names else model_names[0]
        return {
            "required": {
                "inputcount": ("INT", {"default": 6, "min": 1, "max": 24, "step": 1}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "model_name": (model_names, {"default": default_model}),
                "max_tokens": ("INT", {"default": 2048, "min": 1, "max": 65536, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 0.5, "max": 2.0, "step": 0.1}),
                "frame_count": ("INT", {"default": 6, "min": 1, "max": 24, "step": 1}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "video": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("answer", "reasoning", "status")
    FUNCTION = "run"
    CATEGORY = "TDXH/tdxh_api"

    def _get_backend(self):
        if self._backend_node is not None:
            return self._backend_node

        module = _load_qwenvl_module()
        self._backend_node = module.AILab_QwenVL_GGUF_Advanced()
        return self._backend_node

    def run(
        self,
        inputcount,
        prompt,
        system_prompt,
        model_name,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        frame_count,
        keep_model_loaded,
        image_1=None,
        video=None,
        **kwargs,
    ):
        try:
            effective_images = _collect_effective_images(inputcount, image_1, **kwargs)
            provider_name = _provider_name_for_model(model_name)

            has_video = not _is_empty_media_input(video)
            if not effective_images and not has_video:
                return self._text_backend.run(
                    provider_name=provider_name,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    max_tokens=max_tokens,
                    keep_model_loaded=keep_model_loaded,
                )

            single_image = effective_images[0] if effective_images else None
            video_items = []
            if len(effective_images) > 1:
                video_items.extend(effective_images[1:])
            if has_video:
                video_items.append(video)
            combined_video = video_items if video_items else None

            effective_frame_count = int(frame_count)
            if combined_video is None and single_image is not None:
                effective_frame_count = 1

            merged_prompt = _combine_prompt(system_prompt, prompt)
            backend = self._get_backend()
            payload = dict(
                model_name=model_name,
                device="auto",
                preset_prompt="🖼️ Detailed Description",
                custom_prompt=merged_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                frame_count=max(1, effective_frame_count),
                ctx=8192,
                n_batch=256,
                gpu_layers=-1,
                image_max_tokens=4096,
                top_k=0,
                pool_size=4194304,
                keep_model_loaded=keep_model_loaded,
                seed=1,
                image=single_image,
                video=combined_video,
            )
            result = _run_backend_process(backend, payload, CANONICAL_NODE_NAME)
            answer = result[0] if isinstance(result, tuple) and result else ""
            return (str(answer or ""), "", "OK")
        except Exception as exc:
            message = str(exc)
            print(f"[{CANONICAL_NODE_NAME}] ERROR: {message}")
            return ("", "", message)


class _TdxhLocalQwenVLTextBackend:
    def __init__(self):
        self._backend_node = None

    def _get_backend(self):
        if self._backend_node is not None:
            return self._backend_node

        module = _load_qwenvl_module()
        self._backend_node = module.AILab_QwenVL_GGUF_Advanced()
        return self._backend_node

    def run(
        self,
        provider_name,
        prompt,
        system_prompt,
        temperature,
        top_p,
        repetition_penalty,
        max_tokens,
        keep_model_loaded=True,
    ):
        model_name = get_local_qwenvl_model_name(provider_name)
        if not model_name:
            return ("", "", f"Unsupported local QwenVL provider: {provider_name}")

        try:
            merged_prompt = _combine_prompt(system_prompt, prompt)
            backend = self._get_backend()
            payload = dict(
                model_name=model_name,
                device="auto",
                preset_prompt="🖼️ Detailed Description",
                custom_prompt=merged_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                frame_count=1,
                ctx=8192,
                n_batch=256,
                gpu_layers=-1,
                image_max_tokens=4096,
                top_k=0,
                pool_size=4194304,
                keep_model_loaded=keep_model_loaded,
                seed=1,
                image=None,
                video=None,
            )
            result = _run_backend_process(backend, payload, "LocalQwenVLText")
            answer = result[0] if isinstance(result, tuple) and result else ""
            return (str(answer or ""), "", "OK")
        except Exception as exc:
            message = str(exc)
            print(f"[LocalQwenVLText] ERROR: {message}")
            return ("", "", message)

NODE_CLASS_MAPPINGS = {
    CANONICAL_NODE_NAME: TdxhLocalQwenVLDynamicVisionChat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    CANONICAL_NODE_NAME: CANONICAL_NODE_NAME,
}
