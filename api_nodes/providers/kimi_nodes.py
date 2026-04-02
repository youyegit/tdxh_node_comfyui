import base64
import json
import os
import time
import traceback
from urllib.parse import urlparse
from io import BytesIO

import numpy as np
import requests
from PIL import Image


PROVIDERS_DIR = os.path.dirname(os.path.abspath(__file__))
API_NODES_DIR = os.path.dirname(PROVIDERS_DIR)
PLUGIN_ROOT_DIR = os.path.dirname(API_NODES_DIR)
COMFY_ROOT_DIR = os.path.dirname(os.path.dirname(PLUGIN_ROOT_DIR))
CONFIG_PATH = os.path.join(API_NODES_DIR, "configs", "kimi_config.json")
DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"
DEFAULT_TIMEOUT = 60
OVERLOAD_RETRY_COUNT = 3
OVERLOAD_RETRY_DELAYS = (1.0, 2.0)
DEFAULT_PLACEHOLDER_IMAGE_PATH = os.path.join(COMFY_ROOT_DIR, "input", "example.png")
_PLACEHOLDER_IMAGE_HASHES = None
_PLACEHOLDER_IMAGE_SIGNATURE = None


def _load_config():
    config = {}
    if os.path.isfile(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            config = json.load(handle)

    api_key = (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("MOONSHOT_API_KEY")
        or os.environ.get("KIMI_API_KEY")
        or config.get("api_key", "")
    )
    base_url = (
        os.environ.get("ANTHROPIC_BASE_URL")
        or os.environ.get("MOONSHOT_BASE_URL")
        or os.environ.get("KIMI_BASE_URL")
        or config.get("base_url", DEFAULT_BASE_URL)
    )
    timeout = config.get("timeout_seconds", DEFAULT_TIMEOUT)

    return {
        "api_key": str(api_key).strip(),
        "base_url": str(base_url).rstrip("/"),
        "timeout_seconds": int(timeout),
    }


def _is_kimi_coding_base_url(base_url):
    parsed = urlparse(base_url)
    host = parsed.netloc.lower()
    path = parsed.path.lower().rstrip("/")
    return host == "api.kimi.com" and path.startswith("/coding")


def _normalize_base_url(base_url):
    normalized = str(base_url).rstrip("/")
    if _is_kimi_coding_base_url(normalized) and not normalized.lower().endswith("/v1"):
        return normalized + "/v1"
    return normalized


def _normalize_message_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _image_to_data_url(image):
    if image is None:
        raise ValueError("Image input is empty.")

    if hasattr(image, "cpu"):
        image = image.cpu().numpy()

    image = np.asarray(image)
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(image)

    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _image_to_numpy_rgb(image):
    if hasattr(image, "cpu"):
        image = image.cpu().numpy()

    image = np.asarray(image)
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def _average_hash_from_rgb_array(image_array, size=8):
    pil_image = Image.fromarray(image_array).convert("L").resize((size, size), Image.Resampling.LANCZOS)
    arr = np.asarray(pil_image, dtype=np.float32)
    mean_value = float(arr.mean())
    return "".join("1" if value >= mean_value else "0" for value in arr.flatten())


def _signature_from_rgb_array(image_array, size=32):
    pil_image = Image.fromarray(image_array).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    arr = np.asarray(pil_image, dtype=np.float32) / 255.0
    return arr


def _load_placeholder_image_hashes():
    global _PLACEHOLDER_IMAGE_HASHES

    if _PLACEHOLDER_IMAGE_HASHES is not None:
        return _PLACEHOLDER_IMAGE_HASHES

    hashes = set()
    if os.path.isfile(DEFAULT_PLACEHOLDER_IMAGE_PATH):
        try:
            placeholder = Image.open(DEFAULT_PLACEHOLDER_IMAGE_PATH).convert("RGB")
            hashes.add(_average_hash_from_rgb_array(np.asarray(placeholder, dtype=np.uint8)))
        except Exception:
            pass

    _PLACEHOLDER_IMAGE_HASHES = hashes
    return _PLACEHOLDER_IMAGE_HASHES


def _load_placeholder_image_signature():
    global _PLACEHOLDER_IMAGE_SIGNATURE

    if _PLACEHOLDER_IMAGE_SIGNATURE is not None:
        return _PLACEHOLDER_IMAGE_SIGNATURE

    if os.path.isfile(DEFAULT_PLACEHOLDER_IMAGE_PATH):
        try:
            placeholder = Image.open(DEFAULT_PLACEHOLDER_IMAGE_PATH).convert("RGB")
            _PLACEHOLDER_IMAGE_SIGNATURE = _signature_from_rgb_array(np.asarray(placeholder, dtype=np.uint8))
            return _PLACEHOLDER_IMAGE_SIGNATURE
        except Exception:
            pass

    _PLACEHOLDER_IMAGE_SIGNATURE = None
    return _PLACEHOLDER_IMAGE_SIGNATURE


def _is_placeholder_image_tensor(image):
    # Treat ComfyUI's built-in input/example.png as an empty placeholder image
    # even after common resize-like preprocessing, so workflows can keep
    # placeholder LoadImage nodes connected without triggering a real upload.
    hashes = _load_placeholder_image_hashes()

    try:
        image_array = _image_to_numpy_rgb(image)
    except Exception:
        return False

    if hashes:
        image_hash = _average_hash_from_rgb_array(image_array)
        if image_hash in hashes:
            return True

    placeholder_signature = _load_placeholder_image_signature()
    if placeholder_signature is None:
        return False

    try:
        image_signature = _signature_from_rgb_array(image_array)
    except Exception:
        return False

    mae = float(np.mean(np.abs(image_signature - placeholder_signature)))
    return mae <= 0.015


def _build_kimi_user_content(prompt, image_urls):
    content = [{"type": "text", "text": prompt}]
    for image_url in image_urls:
        content.append({"type": "image_url", "image_url": image_url})
    return content


def _is_empty_image_input(image):
    if image is None:
        return True
    if isinstance(image, str):
        return True
    if _is_placeholder_image_tensor(image):
        return True
    return False


def _empty_image_reason(image, idx):
    if image is None:
        return f"image_{idx} is empty"
    if isinstance(image, str):
        value = image.strip() or "<empty string>"
        return f"image_{idx} is empty (placeholder '{value}')"
    if _is_placeholder_image_tensor(image):
        return f"image_{idx} is empty (placeholder image 'example.png')"
    return f"image_{idx} is empty"


def _validate_prompt(prompt, node_label):
    if str(prompt).strip():
        return ""
    return f"{node_label}: request was not sent to Kimi because prompt is empty."


def _collect_image_urls(images, node_label):
    image_urls = []
    gap_found = False
    first_empty_reason = ""

    for idx, image in enumerate(images, start=1):
        if _is_empty_image_input(image):
            gap_found = True
            if not first_empty_reason:
                first_empty_reason = _empty_image_reason(image, idx)
            continue

        if gap_found:
            raise ValueError(
                f"{node_label}: request was not sent to Kimi because {first_empty_reason}, so image_{idx} cannot be used after that. "
                "Only trailing image inputs may be empty."
            )

        try:
            image_urls.append(_image_to_data_url(image))
        except Exception as exc:
            raise ValueError(f"{node_label}: request was not sent to Kimi because image_{idx} could not be encoded: {exc}") from exc

    if not image_urls:
        if first_empty_reason:
            raise ValueError(f"{node_label}: request was not sent to Kimi because no valid image was provided. First empty slot: {first_empty_reason}.")
        raise ValueError(f"{node_label}: request was not sent to Kimi because no valid image was provided.")

    return image_urls


def _should_retry_overload(status_code, error_message):
    if status_code not in (429, 500, 502, 503, 504):
        return False

    lowered = str(error_message).lower()
    overload_markers = (
        "overloaded",
        "try again later",
        "server is busy",
        "service unavailable",
        "temporarily unavailable",
    )
    return any(marker in lowered for marker in overload_markers)


class _KimiBaseNode:
    def __init__(self):
        self.message_history = []

    def _config_error(self):
        return (
            "Kimi API key is missing. Set MOONSHOT_API_KEY/KIMI_API_KEY or create kimi_config.json from "
            "api_nodes/configs/kimi_config.example.json in custom_nodes/tdxh_node_comfyui."
        )

    def _request(self, payload):
        try:
            config = _load_config()
        except Exception as exc:
            return None, f"Failed to load Kimi config: {exc}"

        if not config["api_key"]:
            return None, self._config_error()

        base_url = _normalize_base_url(config["base_url"])
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
        }

        for attempt in range(1, OVERLOAD_RETRY_COUNT + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=config["timeout_seconds"],
                )
            except requests.RequestException as exc:
                return None, f"Kimi request failed: {exc}"

            try:
                data = response.json()
            except ValueError:
                text = response.text[:1000]
                return None, f"Kimi returned non-JSON response ({response.status_code}): {text}"

            if response.ok:
                return data, ""

            error_message = data.get("error", {}).get("message") or data.get("message") or json.dumps(data, ensure_ascii=False)
            if response.status_code == 403 and "currently only available for Coding Agents" in error_message:
                error_message = (
                    "Kimi For Coding rejected this request. The Kimi Code endpoint currently allows only supported coding "
                    "agents such as Kimi CLI / Claude Code / Roo Code. For a ComfyUI custom node, use the Moonshot "
                    "Open Platform endpoint https://api.moonshot.ai/v1 with a Moonshot API key instead."
                )
                return None, f"Kimi API error {response.status_code}: {error_message}"

            if _should_retry_overload(response.status_code, error_message) and attempt < OVERLOAD_RETRY_COUNT:
                delay = OVERLOAD_RETRY_DELAYS[min(attempt - 1, len(OVERLOAD_RETRY_DELAYS) - 1)]
                print(
                    f"[Kimi API] Temporary overload on attempt {attempt}/{OVERLOAD_RETRY_COUNT}. "
                    f"Retrying in {delay:.1f}s. Detail: {error_message}"
                )
                time.sleep(delay)
                continue

            if _should_retry_overload(response.status_code, error_message):
                return None, (
                    f"Kimi API overloaded after {OVERLOAD_RETRY_COUNT} attempts. "
                    f"The request was sent but the server stayed busy: {error_message}"
                )

            if response.status_code == 429:
                error_message = f"Rate limited or account/billing issue: {error_message}"
            return None, f"Kimi API error {response.status_code}: {error_message}"

        return None, "Kimi request failed for an unknown reason."

    def _extract_message(self, response_data):
        choices = response_data.get("choices") or []
        if not choices:
            raise ValueError("No choices found in Kimi response.")

        message = choices[0].get("message") or {}
        reasoning = _normalize_message_content(message.get("reasoning_content", ""))
        answer = _normalize_message_content(message.get("content", ""))
        return reasoning, answer


class TdxhKimiChat(_KimiBaseNode):
    DESCRIPTION = (
        "Text chat node for Kimi/Moonshot. Supports optional history, thinking toggle, "
        "and returns answer, reasoning, and status."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "system_prompt": (
                    "STRING",
                    {"multiline": True, "default": ""},
                ),
                "keep_history": ("BOOLEAN", {"default": False}),
                "clear_history": ("BOOLEAN", {"default": False}),
                "thinking_enabled": ("BOOLEAN", {"default": True}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 65536, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("answer", "reasoning", "status")
    FUNCTION = "run"
    CATEGORY = "TDXH/tdxh_api"

    def run(self, prompt, system_prompt, keep_history, clear_history, thinking_enabled, temperature, max_tokens):
        try:
            config = _load_config()
        except Exception as exc:
            message = f"Failed to load Kimi config: {exc}"
            print(f"[TdxhKimiChat] ERROR: {message}")
            return ("", "", message)
        is_kimi_coding = _is_kimi_coding_base_url(config["base_url"])

        prompt_error = _validate_prompt(prompt, "TdxhKimiChat")
        if prompt_error:
            print(f"[TdxhKimiChat] ERROR: {prompt_error}")
            return ("", "", prompt_error)

        if clear_history:
            self.message_history = []

        if keep_history:
            messages = [{"role": "system", "content": system_prompt}] + list(self.message_history)
        else:
            messages = [{"role": "system", "content": system_prompt}]

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": "kimi-for-coding" if is_kimi_coding else "kimi-k2.5",
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }

        if is_kimi_coding:
            if thinking_enabled:
                payload["reasoning_effort"] = "medium"
        else:
            if thinking_enabled:
                payload["temperature"] = temperature
            else:
                payload["thinking"] = {"type": "disabled"}

        response_data, error = self._request(payload)
        if error:
            print(f"[TdxhKimiChat] ERROR: {error}")
            return ("", "", error)

        try:
            reasoning, answer = self._extract_message(response_data)
        except Exception as exc:
            traceback.print_exc()
            message = f"Failed to parse Kimi response: {exc}"
            print(f"[TdxhKimiChat] ERROR: {message}")
            return ("", "", message)

        if keep_history:
            self.message_history = list(messages[1:])
            assistant_message = {"role": "assistant", "content": answer}
            if reasoning:
                assistant_message["reasoning_content"] = reasoning
            self.message_history.append(assistant_message)

        return (answer, reasoning, "OK")


class TdxhKimiDynamicVisionChat(_KimiBaseNode):
    DESCRIPTION = (
        "Dynamic multi-image Kimi vision chat node. Increase image inputs with 'Update inputs'. "
        "Trailing image slots may be empty, but gaps in the middle are not allowed. "
        "Placeholder filenames such as 'example.png' are treated as empty image inputs. "
        "Outputs originating from LoadImage(example.png) are also treated as empty placeholder images."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inputcount": ("INT", {"default": 2, "min": 1, "max": 24, "step": 1}),
                "image_1": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "system_prompt": (
                    "STRING",
                    {"multiline": True, "default": ""},
                ),
                "keep_history": ("BOOLEAN", {"default": False}),
                "clear_history": ("BOOLEAN", {"default": False}),
                "thinking_enabled": ("BOOLEAN", {"default": True}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 65536, "step": 1}),
            },
            "optional": {
                "image_2": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("answer", "reasoning", "status")
    FUNCTION = "run"
    CATEGORY = "TDXH/tdxh_api"

    def run(
        self,
        inputcount,
        image_1,
        prompt,
        system_prompt,
        keep_history,
        clear_history,
        thinking_enabled,
        temperature,
        max_tokens,
        **kwargs,
    ):
        try:
            config = _load_config()
        except Exception as exc:
            message = f"Failed to load Kimi config: {exc}"
            print(f"[TdxhKimiDynamicVisionChat] ERROR: {message}")
            return ("", "", message)

        is_kimi_coding = _is_kimi_coding_base_url(config["base_url"])
        if is_kimi_coding:
            message = (
                "Kimi dynamic vision node does not support the Kimi Code endpoint. "
                "Use Moonshot Open Platform base_url such as https://api.moonshot.ai/v1 or https://api.moonshot.cn/v1."
            )
            print(f"[TdxhKimiDynamicVisionChat] ERROR: {message}")
            return ("", "", message)

        prompt_error = _validate_prompt(prompt, "TdxhKimiDynamicVisionChat")
        if prompt_error:
            print(f"[TdxhKimiDynamicVisionChat] ERROR: {prompt_error}")
            return ("", "", prompt_error)

        images = [image_1]
        for idx in range(2, int(inputcount) + 1):
            images.append(kwargs.get(f"image_{idx}"))

        try:
            image_urls = _collect_image_urls(images, "TdxhKimiDynamicVisionChat")
        except Exception as exc:
            message = str(exc)
            print(f"[TdxhKimiDynamicVisionChat] ERROR: {message}")
            return ("", "", message)

        if clear_history:
            self.message_history = []

        if keep_history:
            messages = [{"role": "system", "content": system_prompt}] + list(self.message_history)
        else:
            messages = [{"role": "system", "content": system_prompt}]

        user_content = _build_kimi_user_content(prompt, image_urls)
        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": "kimi-k2.5",
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }

        if thinking_enabled:
            payload["temperature"] = temperature
        else:
            payload["thinking"] = {"type": "disabled"}

        response_data, error = self._request(payload)
        if error:
            print(f"[TdxhKimiDynamicVisionChat] ERROR: {error}")
            return ("", "", error)

        try:
            reasoning, answer = self._extract_message(response_data)
        except Exception as exc:
            traceback.print_exc()
            message = f"Failed to parse Kimi dynamic vision response: {exc}"
            print(f"[TdxhKimiDynamicVisionChat] ERROR: {message}")
            return ("", "", message)

        if keep_history:
            self.message_history = list(messages[1:])
            assistant_message = {"role": "assistant", "content": answer}
            if reasoning:
                assistant_message["reasoning_content"] = reasoning
            self.message_history.append(assistant_message)

        return (answer, reasoning, "OK")


NODE_CLASS_MAPPINGS = {
    "TdxhKimiChat": TdxhKimiChat,
    "TdxhKimiDynamicVisionChat": TdxhKimiDynamicVisionChat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TdxhKimiChat": "TdxhKimiChat",
    "TdxhKimiDynamicVisionChat": "TdxhKimiDynamicVisionChat",
}
