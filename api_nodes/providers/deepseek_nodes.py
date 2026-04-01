import json
import os
import traceback

import requests


PROVIDERS_DIR = os.path.dirname(os.path.abspath(__file__))
API_NODES_DIR = os.path.dirname(PROVIDERS_DIR)
CONFIG_PATH = os.path.join(API_NODES_DIR, "configs", "deepseek_config.json")
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_TIMEOUT = 60


def _load_config():
    config = {}
    if os.path.isfile(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            config = json.load(handle)

    api_key = os.environ.get("DEEPSEEK_API_KEY") or config.get("api_key", "")
    base_url = os.environ.get("DEEPSEEK_BASE_URL") or config.get("base_url", DEFAULT_BASE_URL)
    timeout = config.get("timeout_seconds", DEFAULT_TIMEOUT)

    return {
        "api_key": api_key.strip(),
        "base_url": str(base_url).rstrip("/"),
        "timeout_seconds": int(timeout),
    }


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


def _build_error(message):
    print(f"[TdxhDeepSeekChat] ERROR: {message}")
    return ("", "", message)


class _DeepSeekBaseNode:
    def __init__(self):
        self.message_history = []

    def _config_error(self):
        return (
            "DeepSeek API key is missing. Set DEEPSEEK_API_KEY or create deepseek_config.json from "
            "api_nodes/configs/deepseek_config.example.json in custom_nodes/tdxh_node_comfyui."
        )

    def _request(self, payload):
        try:
            config = _load_config()
        except Exception as exc:
            return None, f"Failed to load DeepSeek config: {exc}"

        if not config["api_key"]:
            return None, self._config_error()

        url = f"{config['base_url']}/chat/completions"
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
            return None, f"DeepSeek request failed: {exc}"

        try:
            data = response.json()
        except ValueError:
            text = response.text[:1000]
            return None, f"DeepSeek returned non-JSON response ({response.status_code}): {text}"

        if not response.ok:
            error_message = data.get("error", {}).get("message") or data.get("message") or json.dumps(data, ensure_ascii=False)
            if response.status_code == 429:
                error_message = f"Rate limited or account/billing issue: {error_message}"
            return None, f"DeepSeek API error {response.status_code}: {error_message}"

        return data, ""

    def _extract_message(self, response_data):
        choices = response_data.get("choices") or []
        if not choices:
            raise ValueError("No choices found in DeepSeek response.")

        message = choices[0].get("message") or {}
        answer = _normalize_message_content(message.get("content", ""))
        reasoning = _normalize_message_content(message.get("reasoning_content", ""))
        return answer, reasoning, message


class TdxhDeepSeekChat(_DeepSeekBaseNode):
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
                "thinking_enabled": ("BOOLEAN", {"default": False}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("answer", "reasoning", "status")
    FUNCTION = "run"
    CATEGORY = "TDXH/tdxh_api"

    def run(self, prompt, system_prompt, keep_history, clear_history, thinking_enabled, temperature, top_p, max_tokens):
        if clear_history:
            self.message_history = []

        if keep_history:
            messages = [{"role": "system", "content": system_prompt}] + list(self.message_history)
        else:
            messages = [{"role": "system", "content": system_prompt}]

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": "deepseek-reasoner" if thinking_enabled else "deepseek-chat",
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }

        if not thinking_enabled:
            payload["temperature"] = temperature
            payload["top_p"] = top_p

        response_data, error = self._request(payload)
        if error:
            return _build_error(error)

        try:
            answer, reasoning, _ = self._extract_message(response_data)
        except Exception as exc:
            traceback.print_exc()
            return _build_error(f"Failed to parse DeepSeek response: {exc}")

        if keep_history:
            self.message_history = list(messages[1:])
            assistant_message = {"role": "assistant", "content": answer}
            if reasoning:
                assistant_message["reasoning_content"] = reasoning
            self.message_history.append(assistant_message)

        return (
            answer,
            reasoning,
            "OK",
        )


NODE_CLASS_MAPPINGS = {
    "TdxhDeepSeekChat": TdxhDeepSeekChat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TdxhDeepSeekChat": "TdxhDeepSeekChat",
}
