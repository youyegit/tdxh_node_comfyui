from .deepseek_nodes import TdxhDeepSeekChat
from .kimi_nodes import TdxhKimiChat


PLATFORM_CHOICES = ["deepseek", "kimi", "disabled"]


class TdxhMultiPlatformChat:
    def __init__(self):
        self._providers = {
            "deepseek": TdxhDeepSeekChat(),
            "kimi": TdxhKimiChat(),
        }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "system_prompt": (
                    "STRING",
                    {"multiline": True, "default": ""},
                ),
                "provider_1": (PLATFORM_CHOICES, {"default": "deepseek"}),
                "provider_2": (PLATFORM_CHOICES, {"default": "kimi"}),
                "provider_3": (PLATFORM_CHOICES, {"default": "disabled"}),
                "thinking_enabled": ("BOOLEAN", {"default": False}),
                "keep_history": ("BOOLEAN", {"default": False}),
                "clear_history": ("BOOLEAN", {"default": False}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 2048, "min": 1, "max": 65536, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("answer", "reasoning", "status", "used_provider", "attempt_log")
    FUNCTION = "run"
    CATEGORY = "TDXH/tdxh_api"

    def _ordered_providers(self, provider_1, provider_2, provider_3):
        ordered = []
        for provider in (provider_1, provider_2, provider_3):
            if provider == "disabled":
                continue
            if provider not in ordered:
                ordered.append(provider)
        return ordered

    def _call_provider(
        self,
        provider_name,
        prompt,
        system_prompt,
        keep_history,
        clear_history,
        thinking_enabled,
        temperature,
        top_p,
        max_tokens,
    ):
        if provider_name == "deepseek":
            return self._providers["deepseek"].run(
                prompt,
                system_prompt,
                keep_history,
                clear_history,
                thinking_enabled,
                temperature,
                top_p,
                max_tokens,
            )

        if provider_name == "kimi":
            return self._providers["kimi"].run(
                prompt,
                system_prompt,
                keep_history,
                clear_history,
                thinking_enabled,
                temperature,
                max_tokens,
            )

        return ("", "", f"Unsupported provider: {provider_name}")

    def run(
        self,
        prompt,
        system_prompt,
        provider_1,
        provider_2,
        provider_3,
        thinking_enabled,
        keep_history,
        clear_history,
        temperature,
        top_p,
        max_tokens,
    ):
        ordered = self._ordered_providers(provider_1, provider_2, provider_3)
        if not ordered:
            message = "No provider selected. Set provider_1, provider_2, or provider_3 to deepseek or kimi."
            print(f"[TdxhMultiPlatformChat] ERROR: {message}")
            raise RuntimeError(message)

        attempts = []
        should_clear = clear_history

        for provider_name in ordered:
            answer, reasoning, status = self._call_provider(
                provider_name,
                prompt,
                system_prompt,
                keep_history,
                should_clear,
                thinking_enabled,
                temperature,
                top_p,
                max_tokens,
            )

            provider_status = status or ""
            attempts.append(f"{provider_name}: {provider_status}")

            if provider_status == "OK":
                attempt_log = " | ".join(attempts)
                return (answer, reasoning, "OK", provider_name, attempt_log)

            should_clear = False

        final_status = "All providers failed. " + " | ".join(attempts)
        print(f"[TdxhMultiPlatformChat] ERROR: {final_status}")
        raise RuntimeError(final_status)


NODE_CLASS_MAPPINGS = {
    "TdxhMultiPlatformChat": TdxhMultiPlatformChat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TdxhMultiPlatformChat": "TdxhMultiPlatformChat",
}
