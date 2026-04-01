import json
import os
import traceback
from urllib.parse import urlparse

import requests


PROVIDERS_DIR = os.path.dirname(os.path.abspath(__file__))
API_NODES_DIR = os.path.dirname(PROVIDERS_DIR)
CONFIG_PATH = os.path.join(API_NODES_DIR, "configs", "kimi_config.json")
DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"
DEFAULT_TIMEOUT = 60


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

        if not response.ok:
            error_message = data.get("error", {}).get("message") or data.get("message") or json.dumps(data, ensure_ascii=False)
            if response.status_code == 403 and "currently only available for Coding Agents" in error_message:
                error_message = (
                    "Kimi For Coding rejected this request. The Kimi Code endpoint currently allows only supported coding "
                    "agents such as Kimi CLI / Claude Code / Roo Code. For a ComfyUI custom node, use the Moonshot "
                    "Open Platform endpoint https://api.moonshot.ai/v1 with a Moonshot API key instead."
                )
            elif response.status_code == 429:
                error_message = f"Rate limited or account/billing issue: {error_message}"
            return None, f"Kimi API error {response.status_code}: {error_message}"

        return data, ""

    def _extract_message(self, response_data):
        choices = response_data.get("choices") or []
        if not choices:
            raise ValueError("No choices found in Kimi response.")

        message = choices[0].get("message") or {}
        reasoning = _normalize_message_content(message.get("reasoning_content", ""))
        answer = _normalize_message_content(message.get("content", ""))
        return reasoning, answer


class TdxhKimiChat(_KimiBaseNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "system_prompt": (
                    "STRING",
                    {"multiline": True, "default": "You are a helpful assistant."},
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


NODE_CLASS_MAPPINGS = {
    "TdxhKimiChat": TdxhKimiChat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TdxhKimiChat": "TdxhKimiChat",
}
