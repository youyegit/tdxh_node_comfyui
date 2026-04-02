from .deepseek_nodes import TdxhDeepSeekChat
from .kimi_nodes import TdxhKimiChat, TdxhKimiDynamicVisionChat, _is_empty_image_input


DYNAMIC_VISION_PLATFORM_CHOICES = ["kimi", "deepseek", "disabled"]


class TdxhMultiPlatformDynamicVisionChat:
    DESCRIPTION = (
        "Dynamic multi-provider vision chat node with provider fallback. "
        "Providers are tried in provider_1 -> provider_2 -> provider_3 order. "
        "If no effective image is provided, the node automatically falls back to text chat using the same provider order. "
        "Trailing image slots may be empty, but gaps in the middle are not allowed. "
        "Placeholder filenames such as 'example.png' are treated as empty image inputs. "
        "Outputs originating from LoadImage(example.png) are also treated as empty placeholder images."
    )

    def __init__(self):
        self._vision_providers = {
            "kimi": TdxhKimiDynamicVisionChat(),
        }
        self._text_providers = {
            "deepseek": TdxhDeepSeekChat(),
            "kimi": TdxhKimiChat(),
        }

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
                "provider_1": (DYNAMIC_VISION_PLATFORM_CHOICES, {"default": "kimi"}),
                "provider_2": (DYNAMIC_VISION_PLATFORM_CHOICES, {"default": "disabled"}),
                "provider_3": (DYNAMIC_VISION_PLATFORM_CHOICES, {"default": "disabled"}),
                "thinking_enabled": ("BOOLEAN", {"default": False}),
                "keep_history": ("BOOLEAN", {"default": False}),
                "clear_history": ("BOOLEAN", {"default": False}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 65536, "step": 1}),
            },
            "optional": {
                "image_2": ("IMAGE",),
            },
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
        if provider_name == "kimi":
            return self._vision_providers["kimi"].run(
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
            )

        if provider_name == "deepseek":
            return (
                "",
                "",
                "DeepSeek vision fallback is not available yet. As of April 2, 2026, DeepSeek official API docs do not document public image input support for the Open Platform API.",
            )

        return ("", "", f"Unsupported provider: {provider_name}")

    def _call_text_provider(
        self,
        provider_name,
        prompt,
        system_prompt,
        keep_history,
        clear_history,
        thinking_enabled,
        temperature,
        max_tokens,
    ):
        if provider_name == "deepseek":
            return self._text_providers["deepseek"].run(
                prompt,
                system_prompt,
                keep_history,
                clear_history,
                thinking_enabled,
                temperature,
                1.0,
                max_tokens,
            )

        if provider_name == "kimi":
            return self._text_providers["kimi"].run(
                prompt,
                system_prompt,
                keep_history,
                clear_history,
                thinking_enabled,
                temperature,
                max_tokens,
            )

        return ("", "", f"Unsupported text fallback provider: {provider_name}")

    def _has_any_effective_image(self, inputcount, image_1, **kwargs):
        images = [image_1]
        for idx in range(2, int(inputcount) + 1):
            images.append(kwargs.get(f"image_{idx}"))
        return any(not _is_empty_image_input(image) for image in images)

    def run(
        self,
        inputcount,
        image_1,
        prompt,
        system_prompt,
        provider_1,
        provider_2,
        provider_3,
        thinking_enabled,
        keep_history,
        clear_history,
        temperature,
        max_tokens,
        **kwargs,
    ):
        ordered = self._ordered_providers(provider_1, provider_2, provider_3)
        if not ordered:
            message = "No provider selected. Set provider_1, provider_2, or provider_3 to a visual provider such as kimi."
            print(f"[TdxhMultiPlatformDynamicVisionChat] ERROR: {message}")
            raise RuntimeError(message)

        if not self._has_any_effective_image(inputcount, image_1, **kwargs):
            attempts = []
            should_clear = clear_history

            for provider_name in ordered:
                answer, reasoning, status = self._call_text_provider(
                    provider_name,
                    prompt,
                    system_prompt,
                    keep_history,
                    should_clear,
                    thinking_enabled,
                    temperature,
                    max_tokens,
                )

                provider_status = status or ""
                attempts.append(f"{provider_name}[text]: {provider_status}")

                if provider_status == "OK":
                    attempt_log = " | ".join(attempts)
                    return (answer, reasoning, "OK", f"{provider_name}[text]", attempt_log)

                should_clear = False

            final_status = "All text fallback providers failed. " + " | ".join(attempts)
            print(f"[TdxhMultiPlatformDynamicVisionChat] ERROR: {final_status}")
            raise RuntimeError(final_status)

        attempts = []
        should_clear = clear_history

        for provider_name in ordered:
            answer, reasoning, status = self._call_provider(
                provider_name,
                inputcount,
                image_1,
                prompt,
                system_prompt,
                keep_history,
                should_clear,
                thinking_enabled,
                temperature,
                max_tokens,
                **kwargs,
            )

            provider_status = status or ""
            attempts.append(f"{provider_name}: {provider_status}")

            if provider_status == "OK":
                attempt_log = " | ".join(attempts)
                return (answer, reasoning, "OK", provider_name, attempt_log)

            should_clear = False

        final_status = "All dynamic visual providers failed. " + " | ".join(attempts)
        print(f"[TdxhMultiPlatformDynamicVisionChat] ERROR: {final_status}")
        raise RuntimeError(final_status)


NODE_CLASS_MAPPINGS = {
    "TdxhMultiPlatformDynamicVisionChat": TdxhMultiPlatformDynamicVisionChat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TdxhMultiPlatformDynamicVisionChat": "TdxhMultiPlatformDynamicVisionChat",
}
