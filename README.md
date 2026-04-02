# Introduction
Some nodes for stable diffusion comfyui.Sometimes it helps conveniently to use less nodes for doing the same things.

If you use workflow in my "blogs" repo, you need to dowmload these nodes.I don't guarantee that the nodes will stay the same always. Some nodes maybe have been changed if you update the new version.
# How to install
## The repo
The same with others custom nodes. Just cd custom_nodes and then git clone.
## Translator model
If you use prompt translator to translate Chinese to English offline, you need download some models.
Download the translator models from https://huggingface.co/facebook/mbart-large-50-many-to-one-mmt/tree/main into folder named "model" of this repo.
The model folder tree of this repo:
model/
└── mbart-large-50-many-to-many-mmt__only_to_English/
    ├── pytorch_model.bin
    ├── config.json
    ├── sentencepiece.bpe.model
    ├── special_tokens_map.json
    ├── tmp2l0rt359
    └── tokenizer_config.json
## Environments
cd (this repo)
pip install -r requirements.txt

# Nodes Introductions
## tdxh_image
- "TdxhImageToSize": TdxhImageToSize can Convert an image to size.
- "TdxhImageToSizeAdvanced":TdxhImageToSizeAdvanced can Convert an image to size and it will let you choose what to follow:"only_width", "only_height", "both_width_and_height","width * height", "only_ratio","only_image","get_SDXL_best_size".
## tdxh_model
- "TdxhLoraLoader": TdxhLoraLoader adds a switch to the "LoraLoader", which shows as "bool_int" (0 -> OFF, 1 -> ON), and lets you choose "only_strength_both" or "strength_model_and_strength_clip".
## tdxh_data
- "TdxhIntInput": TdxhIntInput returns the "INT" type.
- "TdxhFloatInput": TdxhFloatInput returns the "FLOAT" type.
- "TdxhStringInput": TdxhStringInput returns the "STRING" type.
- "TdxhSaveText": TdxhSaveText saves input text to an output text file and returns both the input text and the saved path as "STRING" outputs.
- "TdxhStringInputTranslator": TdxhStringInputTranslator returns the translated "STRING" type. (You need to download the translator model.)
## tdxh_bool
- "TdxhOnOrOff": TdxhOnOrOff returns the "NUMBER" and "INT" type. When switching to ON, it returns 1; when switching to OFF, it returns 0.
- "TdxhBoolNumber": TdxhBoolNumber is similar to TdxhOnOrOff but lets you choose what to follow. "control_by_master" is the main control: if OFF, it returns "bool_int"; if ON, then only when both "bool_int_from_master" and "bool_int" are 1 does it return 1.
## tdxh_efficiency
- "TdxhClipVison": TdxhClipVison adds a switch to the "CLIPVisionLoader" and the "clip_vision".
- "TdxhControlNetProcessor": TdxhControlNetProcessor adds a switch to ControlNet nodes and lets you preprocess the image. (It needs AUX preprocessor nodes.)
- "TdxhControlNetApply": TdxhControlNetApply adds a switch to ControlNet nodes and makes them more efficient to use.
- "TdxhReference": TdxhReference makes the "reference_only" node more efficient.
- "TdxhImg2ImgLatent": TdxhImg2ImgLatent can switch between original(main) latent and image latent (OFF -> main latent, ON -> image latent).

## API nodes
### DeepSeek API nodes
This repo now includes:
- `TdxhDeepSeekChat`

Config priority:
1. environment variable `DEEPSEEK_API_KEY`
2. local file `deepseek_config.json`

You can create `deepseek_config.json` from `deepseek_config.example.json`:

```json
{
  "api_key": "sk-your-deepseek-api-key",
  "base_url": "https://api.deepseek.com",
  "timeout_seconds": 60
}
```

Notes:
- `TdxhDeepSeekChat` outputs `answer`, `reasoning`, `status`
- `TdxhDeepSeekChat` now has a `thinking_enabled` toggle, switching between `deepseek-chat` and `deepseek-reasoner`
- both nodes support optional multi-round history with `keep_history`
- `clear_history` clears the stored conversation state inside the node instance
- config files are stored under `api_nodes/configs/`

### Kimi API nodes
This repo now also includes:
- `TdxhKimiChat`
- `TdxhKimiDynamicVisionChat`

Config priority:
1. environment variable `MOONSHOT_API_KEY`
2. environment variable `KIMI_API_KEY`
3. local file `kimi_config.json`

You can create `kimi_config.json` from `kimi_config.example.json`:

```json
{
  "api_key": "sk-your-moonshot-api-key",
  "base_url": "https://api.moonshot.ai/v1",
  "timeout_seconds": 60
}
```

Notes:
- `TdxhKimiChat` uses `kimi-k2.5`
- `TdxhKimiChat` can disable thinking by sending `thinking: {"type":"disabled"}`
- `TdxhKimiChat` already has a `thinking_enabled` toggle in the node UI
- `TdxhKimiDynamicVisionChat` supports a dynamic number of image inputs with an `Update inputs` button
- dynamic image inputs allow trailing image inputs to be empty, but do not allow gaps in the middle; if `image_4` is connected then `image_1` to `image_3` must also be connected
- `TdxhKimiDynamicVisionChat` requires Moonshot Open Platform endpoints, not the Kimi Code endpoint
- ComfyUI placeholder images coming from `LoadImage(example.png)` are treated as empty image inputs, including common resize-like preprocessing results
- both nodes output `reasoning_content` when the model returns it
- if `keep_history` is enabled, the node stores `reasoning_content` in assistant history to follow Moonshot's thinking-model guidance
- config files are stored under `api_nodes/configs/`

### Multi-platform fallback node
This repo also includes:
- `TdxhMultiPlatformChat`
- `TdxhMultiPlatformDynamicVisionChat`

Features:
- supports provider priority ordering with `provider_1`, `provider_2`, and `provider_3`
- `provider_3` defaults to `disabled` so the node structure can stay stable for future expansion
- current providers: `deepseek`, `kimi`
- reserved for future extension by adding more providers
- automatically falls back to the next provider when the previous one returns a non-OK status
- outputs:
  - `answer`
  - `reasoning`
  - `status`
  - `used_provider`
  - `attempt_log`

Vision notes:
- `TdxhMultiPlatformDynamicVisionChat` supports dynamic image inputs with an `Update inputs` button
- if all connected images are placeholders such as `LoadImage(example.png)`, the node treats them as empty and falls back to text chat
- `TdxhMultiPlatformDynamicVisionChat` currently works with `kimi`; `deepseek` is kept as a reserved provider slot but returns an explicit unsupported error until DeepSeek publishes official public vision API documentation

# Thanks
Some codes are from The official [ComfyUI](https://github.com/comfyanonymous/ComfyUI.git) and other custom nodes like The [was-node-suite-comfyui](https://github.com/WASasquatch/was-node-suite-comfyui.git).
The translator's main code is from [prompt_translator](https://github.com/ParisNeo/prompt_translator.git).
