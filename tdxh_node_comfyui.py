import json
from PIL import Image
import numpy as np
import os
import sys
import torch

# sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import comfy.utils
import comfy.sd
import comfy.model_management as model_management

import folder_paths

from .tdxh_lib import get_SDXL_best_size, target_sizes_show

# Get the absolute path of various directories
my_dir = os.path.dirname(os.path.abspath(__file__))
custom_nodes_dir = os.path.abspath(os.path.join(my_dir, '..'))
comfy_dir = os.path.abspath(os.path.join(my_dir, '..', '..'))
sys.path.append(my_dir)

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def _tdxh_is_empty_image_input(image):
    if image is None:
        return True
    if isinstance(image, str):
        return True
    if not hasattr(image, "shape"):
        return True
    shape = tuple(image.shape)
    if len(shape) < 3:
        return True
    if len(shape) == 4 and shape[0] == 0:
        return True
    return False


def _tdxh_collect_effective_images(inputcount, image_1=None, **kwargs):
    images = [image_1]
    for idx in range(2, int(inputcount) + 1):
        images.append(kwargs.get(f"image_{idx}"))
    return [image for image in images if not _tdxh_is_empty_image_input(image)]


def _tdxh_collect_effective_media(inputcount, prefix, first_value=None, validator=None, **kwargs):
    values = [first_value]
    for idx in range(2, int(inputcount) + 1):
        values.append(kwargs.get(f"{prefix}_{idx}"))
    if validator is None:
        validator = lambda value: value is not None
    return [value for value in values if validator(value)]


def _tdxh_is_valid_audio_input(audio):
    return isinstance(audio, dict) and audio.get("waveform") is not None


def _tdxh_image_shape(image):
    if _tdxh_is_empty_image_input(image):
        return (0, 0, 0)
    shape = tuple(image.shape)
    if len(shape) == 4:
        frames, height, width = int(shape[0]), int(shape[1]), int(shape[2])
        return (frames, width, height)
    if len(shape) == 3:
        height, width = int(shape[0]), int(shape[1])
        return (1, width, height)
    return (0, 0, 0)


def _tdxh_audio_stats(audio):
    if not isinstance(audio, dict):
        return (0.0, 0, 0)
    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate") or 0)
    if waveform is None or sample_rate <= 0 or not hasattr(waveform, "shape"):
        return (0.0, sample_rate, 0)
    channels = int(waveform.shape[1]) if len(waveform.shape) >= 2 else 0
    samples = int(waveform.shape[-1]) if len(waveform.shape) >= 1 else 0
    duration_seconds = float(samples / sample_rate) if sample_rate > 0 else 0.0
    return (duration_seconds, sample_rate, channels)


def _tdxh_trim_audio_to_seconds(audio, target_seconds):
    if not isinstance(audio, dict):
        return None
    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate") or 0)
    if waveform is None or sample_rate <= 0 or not hasattr(waveform, "shape"):
        return audio
    target_samples = max(int(round(float(target_seconds) * sample_rate)), 0)
    trimmed = waveform[..., : min(target_samples, int(waveform.shape[-1]))]
    return {
        "waveform": trimmed,
        "sample_rate": sample_rate,
    }


def _tdxh_pick_first(items):
    return items[0] if items else None


class TdxhImageToSize:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("INT","INT","FLOAT","FLOAT","STRING","STRING","NUMBER","NUMBER")
    RETURN_NAMES = ("width_INT", "height_INT","width_FLOAT", "height_FLOAT","width_STRING", "height_STRING","width_NUMBER", "height_NUMBER")
    FUNCTION = "tdxh_image_to_size"
    #OUTPUT_NODE = False
    CATEGORY = "TDXH/tdxh_image"

    def tdxh_image_to_size(self, image):
        image = tensor2pil(image)
        if image.size:
            w, h = image.size[0], image.size[1]
        else:
            w, h = 0, 0
        return self.tdxh_size_out(w,h)
    
    def tdxh_size_out(self,w,h):
        return (w, h, float(w), float(h), str(w), str(h), w, h)

class TdxhImageToSizeAdvanced:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {
                    "default": 768, 
                    "min": 128, 
                    "max": 8192, 
                    "step": 8 
                }),
                "height": ("INT", {
                    "default": 768, 
                    "min": 128, 
                    "max": 8192, 
                    "step": 8 
                }),
                "width_multiply_by_height": (target_sizes_show,{"default": '1.0:(1024, 1024)'}),
                "ratio": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 10.0, 
                    "step": 0.1
                }),
                "what_to_follow": ([
                    "only_width", "only_height", "both_width_and_height","width * height", "only_ratio", 
                    "only_image","get_SDXL_best_size"
                ],),
            }
        }

    RETURN_TYPES = ("INT","INT","FLOAT","FLOAT","STRING","STRING","NUMBER","NUMBER")
    RETURN_NAMES = ("width_INT", "height_INT","width_FLOAT", "height_FLOAT","width_STRING", "height_STRING","width_NUMBER", "height_NUMBER")
    FUNCTION = "tdxh_image_to_size_advanced"
    #OUTPUT_NODE = False
    CATEGORY = "TDXH/tdxh_image"

    def tdxh_image_to_size_advanced(self, image, width, height, width_multiply_by_height,ratio,what_to_follow):
        image_size = self.tdxh_image_to_size(image)
        # width = self.tdxh_nearest_divisible_by_8(width)
        # height = self.tdxh_nearest_divisible_by_8(height)
        if what_to_follow == "only_image":
            return image_size
        elif what_to_follow == "get_SDXL_best_size":
            w, h = get_SDXL_best_size((image_size[0],image_size[1]))
        elif what_to_follow == "only_ratio":
            w, h = ratio * image_size[0], ratio * image_size[1]
            w, h = self.tdxh_nearest_divisible_by_8(w), self.tdxh_nearest_divisible_by_8(h)
        elif what_to_follow == "both_width_and_height":
            w, h = width, height
        elif  what_to_follow == "width * height":
            w_h_str = width_multiply_by_height.split(':')[-1].strip('()')  # '3.0: (1728, 576)'
            w, h = map(int, w_h_str.split(','))
        elif what_to_follow == "only_width":
            new_height = self.tdxh_nearest_divisible_by_8(image_size[1] * width / image_size[0])
            w, h = width, new_height
        elif what_to_follow == "only_height":
            new_width = self.tdxh_nearest_divisible_by_8(image_size[0] * height / image_size[1])
            w, h = new_width, height

        return self.tdxh_size_out(w,h)
    
    def tdxh_image_to_size(self, image):
        image = tensor2pil(image)
        if image.size:
            w, h = image.size[0], image.size[1]
        else:
            w, h = 0, 0
        return self.tdxh_size_out(w,h)
    
    def tdxh_size_out(self,w,h):
        return (w, h, float(w), float(h), str(w), str(h), w, h)
        
    
    def tdxh_nearest_divisible_by_8(self,num):
        num = round(num)
        remainder = num % 8
        if remainder <= 4:
            return num - remainder
        else:
            return num + (8 - remainder)

# allow setting enable or disable. allow setting strength synchronously
class TdxhLoraLoader:
    def __init__(self):
        self.loaded_lora = None
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "bool_int": ("INT", {"default": 1, "min": 0, "max": 1, "step": 1}),

            "model": ("MODEL",),
            "clip": ("CLIP", ),
            "lora_name": (folder_paths.get_filename_list("loras"), ),
            "strength_model": ("FLOAT", {
                "default": 0.5, "min": -10.0, 
                "max": 10.0, "step": 0.05
                }),
            "strength_clip": ("FLOAT", {
                "default": 0.5, "min": -10.0, 
                "max": 10.0, "step": 0.05
                }),

            "strength_both": ("FLOAT", {
                "default": 0.5, "min": -10.0, 
                "max": 10.0, "step": 0.05
                }),
            "what_to_follow": ([
                "only_strength_both", 
                "strength_model_and_strength_clip"
                ],),
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"

    CATEGORY = "TDXH/tdxh_model"

    def load_lora(self, model, clip, bool_int, lora_name, strength_both,strength_model, strength_clip, what_to_follow):
        from nodes import LoraLoader
        if bool_int == 0:
            return (model, clip)
        if what_to_follow == "only_strength_both":
            strength_model, strength_clip = strength_both, strength_both
        return LoraLoader().load_lora( model, clip, lora_name, strength_model, strength_clip) 

class TdxhVAELoader:
    video_taes = ["taehv", "lighttaew2_2", "lighttaew2_1", "lighttaehy1_5"]
    image_taes = ["taesd", "taesdxl", "taesd3", "taef1"]

    @staticmethod
    def vae_list():
        vaes = folder_paths.get_filename_list("vae")
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        sdxl_taesd_enc = False
        sdxl_taesd_dec = False
        sd1_taesd_enc = False
        sd1_taesd_dec = False
        sd3_taesd_enc = False
        sd3_taesd_dec = False
        f1_taesd_enc = False
        f1_taesd_dec = False

        for v in approx_vaes:
            if v.startswith("taesd_decoder."):
                sd1_taesd_dec = True
            elif v.startswith("taesd_encoder."):
                sd1_taesd_enc = True
            elif v.startswith("taesdxl_decoder."):
                sdxl_taesd_dec = True
            elif v.startswith("taesdxl_encoder."):
                sdxl_taesd_enc = True
            elif v.startswith("taesd3_decoder."):
                sd3_taesd_dec = True
            elif v.startswith("taesd3_encoder."):
                sd3_taesd_enc = True
            elif v.startswith("taef1_encoder."):
                f1_taesd_dec = True
            elif v.startswith("taef1_decoder."):
                f1_taesd_enc = True
            else:
                for tae in TdxhVAELoader.video_taes:
                    if v.startswith(tae):
                        vaes.append(v)

        if sd1_taesd_dec and sd1_taesd_enc:
            vaes.append("taesd")
        if sdxl_taesd_dec and sdxl_taesd_enc:
            vaes.append("taesdxl")
        if sd3_taesd_dec and sd3_taesd_enc:
            vaes.append("taesd3")
        if f1_taesd_dec and f1_taesd_enc:
            vaes.append("taef1")
        vaes.append("pixel_space")
        return vaes

    @staticmethod
    def load_taesd(name):
        sd = {}
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        encoder = next(filter(lambda a: a.startswith(f"{name}_encoder."), approx_vaes))
        decoder = next(filter(lambda a: a.startswith(f"{name}_decoder."), approx_vaes))

        enc = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", encoder))
        for k in enc:
            sd[f"taesd_encoder.{k}"] = enc[k]

        dec = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", decoder))
        for k in dec:
            sd[f"taesd_decoder.{k}"] = dec[k]

        if name == "taesd":
            sd["vae_scale"] = torch.tensor(0.18215)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesdxl":
            sd["vae_scale"] = torch.tensor(0.13025)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesd3":
            sd["vae_scale"] = torch.tensor(1.5305)
            sd["vae_shift"] = torch.tensor(0.0609)
        elif name == "taef1":
            sd["vae_scale"] = torch.tensor(0.3611)
            sd["vae_shift"] = torch.tensor(0.1159)
        return sd

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": (cls.vae_list(),),
                "device": (["main_device", "cpu"],),
                "weight_dtype": (["bf16", "fp16", "fp32"],),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "TDXH/tdxh_model"

    def load_vae(self, vae_name, device, weight_dtype):
        metadata = None
        audio_vae_prefixed = False
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[weight_dtype]

        if device == "main_device":
            device = model_management.get_torch_device()
        elif device == "cpu":
            device = torch.device("cpu")

        if vae_name == "pixel_space":
            sd = {"pixel_space_vae": torch.tensor(1.0)}
        elif vae_name in self.image_taes:
            sd = self.load_taesd(vae_name)
        else:
            model_folder = "vae_approx" if os.path.splitext(vae_name)[0] in self.video_taes else "vae"
            vae_path = folder_paths.get_full_path_or_raise(model_folder, vae_name)
            sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
            if "audio_vae.decoder.conv_in.conv.weight" in sd or any(k.startswith("audio_vae.") for k in sd.keys()):
                audio_vae_prefixed = True

        if audio_vae_prefixed or "vocoder.conv_post.weight" in sd:
            from comfy.ldm.lightricks.vae.audio_vae import AudioVAE
            vae = AudioVAE(sd, metadata)
        else:
            vae = comfy.sd.VAE(sd=sd, device=device, dtype=dtype, metadata=metadata)
            vae.throw_exception_if_invalid()

        return (vae,)

class TdxhIntInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "int_value": ("INT", {
                    "default": 1, 
                    "min": -100000, 
                    "max": 100000, 
                    "step": 1 
                }),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("INT",)
    FUNCTION = "tdxh_value_output"
    #OUTPUT_NODE = False
    CATEGORY = "TDXH/tdxh_data"

    def tdxh_value_output(self,int_value):
        return (int_value,)

class TdxhFloatInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "float_value": ("FLOAT", {
                    "default": 1.0, 
                    "min": -100000.0, 
                    "max": 100000.0, 
                    "step": 0.01
                }),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("FLOAT", )
    FUNCTION = "tdxh_value_output"
    #OUTPUT_NODE = False
    CATEGORY = "TDXH/tdxh_data"

    def tdxh_value_output(self,float_value):
        return (float_value,)

class TdxhStringInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string_value": ("STRING", {
                    "multiline": False, 
                    "default": "tdxh"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "tdxh_value_output"
    #OUTPUT_NODE = False
    CATEGORY = "TDXH/tdxh_data"

    def tdxh_value_output(self, string_value):
        return (string_value,)   

class TdxhSaveText:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "forceInput": True
                }),
                "filename_prefix": ("STRING", {
                    "default": "text/ComfyUI"
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "saved_text_path")
    FUNCTION = "save_text"
    OUTPUT_NODE = True
    CATEGORY = "TDXH/tdxh_data"

    def save_text(self, text, filename_prefix="text/ComfyUI", prompt=None, extra_pnginfo=None):
        if isinstance(text, (list, tuple)):
            text = "\n".join("" if t is None else str(t) for t in text)
        elif text is None:
            text = ""
        else:
            text = str(text)

        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir
        )

        file = f"{filename}_{counter:05}_.txt"
        file_path = os.path.join(full_output_folder, file)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

        saved_path = os.path.join(subfolder, file) if subfolder else file
        return {
            "ui": {
                "text": (saved_path,)
            },
            "result": (text, saved_path)
        }

class TdxhStringInputTranslator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string_value": (
                    "STRING", 
                    {
                        "multiline": True, 
                        "default": "moon"
                    }
                ),
                "bool_int": ("INT", {
                    "default": 1, 
                    "min": 0, 
                    "max": 1, 
                    "step": 1 
                }),
                "input_language": (
                    [
                        r"中文", 
                        r"عربية", 
                        r"Deutsch", 
                        r"Español", 
                        r"Français", 
                        r"हिन्दी", 
                        r"Italiano", 
                        r"日本語", 
                        r"한국어", 
                        r"Português", 
                        r"Русский", 
                        r"Afrikaans", 
                        r"বাংলা", 
                        r"Bosanski", 
                        r"Català", 
                        r"Čeština", 
                        r"Dansk", 
                        r"Ελληνικά", 
                        r"Eesti", 
                        r"فارسی", 
                        r"Suomi", 
                        r"ગુજરાતી", 
                        r"עברית", 
                        r"हिन्दी", 
                        r"Hrvatski", 
                        r"Magyar", 
                        r"Bahasa Indonesia", 
                        r"Íslenska", 
                        r"Javanese", 
                        r"ქართული", 
                        r"Қазақ", 
                        r"ខ្មែរ", 
                        r"ಕನ್ನಡ", 
                        r"한국어", 
                        r"ລາວ", 
                        r"Lietuvių", 
                        r"Latviešu", 
                        r"Македонски", 
                        r"മലയാളം", 
                        r"मराठी", 
                        r"Bahasa Melayu", 
                        r"नेपाली", 
                        r"Nederlands", 
                        r"Norsk", 
                        r"Polski",
                        r"Română", 
                        r"සිංහල", 
                        r"Slovenčina", 
                        r"Slovenščina", 
                        r"Shqip",  
                        r"Turkish", 
                        r"Tiếng Việt",
                    ],
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "tdxh_value_output"
    #OUTPUT_NODE = False
    CATEGORY = "TDXH/tdxh_data"

    def tdxh_value_output(self, string_value, bool_int, input_language):
        if bool_int == 0:
            return (string_value,)
        from tdxh_translator import Prompt,TranslatorScript
        prompt_list=[str(string_value)]
        p_in = Prompt(prompt_list, [""])

        translator = TranslatorScript()
        translator.set_active()
        translator.process(p_in,input_language)
        
        string_value_out=p_in.positive_prompt_list[0] if p_in.positive_prompt_list is not None else ""
        return (string_value_out,)
    
class TdxhOnOrOff:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ON_or_OFF": (["ON", "OFF"],),
            }
        }

    RETURN_TYPES = ("NUMBER","INT")
    RETURN_NAMES = ("NUMBER","INT")
    FUNCTION = "tdxh_value_output"
    #OUTPUT_NODE = False
    CATEGORY = "TDXH/tdxh_bool"

    def tdxh_value_output(self, ON_or_OFF):
        bool_num = 1 if ON_or_OFF == "ON" else 0
        return (bool_num, bool_num)
    
class TdxhBoolNumber:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bool_int": ("INT", {"default": 1, "min": 0, "max": 1, "step": 1}),
                "bool_int_from_master": ("INT", {"default": 1, "min": 0, "max": 1, "step": 1}),
                "control_by_master": (["ON", "OFF"],{"default":"OFF"}),
            }
        }

    RETURN_TYPES = ("NUMBER","INT")
    RETURN_NAMES = ("NUMBER","INT")
    FUNCTION = "tdxh_value_output"
    #OUTPUT_NODE = False
    CATEGORY = "TDXH/tdxh_bool"

    def tdxh_value_output(self, bool_int, bool_int_from_master, control_by_master):
        if control_by_master == "OFF":
            bool_num = bool_int
            return (bool_num, bool_num)
        else:
            bool_num = int(bool(bool_int) and bool(bool_int_from_master))
            return (bool_num, bool_num) 
        

class TdxhToggleMaster:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "boolean": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("BOOLEAN","NUMBER","INT")
    RETURN_NAMES = ("BOOLEAN","NUMBER","INT")
    FUNCTION = "tdxh_value_output"
    #OUTPUT_NODE = False
    CATEGORY = "TDXH/tdxh_bool"

    def tdxh_value_output(self, boolean):
        bool_num = 1 if boolean else 0
        return (boolean, bool_num, bool_num) 

class TdxhToggleGuest:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "boolean": ("BOOLEAN", {"default": True}),
                "boolean_from_master": ("BOOLEAN", {"default": True}),
                "control_by_master": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("BOOLEAN","NUMBER","INT")
    RETURN_NAMES = ("BOOLEAN","NUMBER","INT")
    FUNCTION = "tdxh_value_output"
    #OUTPUT_NODE = False
    CATEGORY = "TDXH/tdxh_bool"

    def tdxh_value_output(self, boolean, boolean_from_master, control_by_master):
        if not control_by_master:
            bool_num = int(boolean)
            return (boolean, bool_num, bool_num) 
        else:
            bool_num = int(boolean and boolean_from_master)
            return (bool(bool_num), bool_num, bool_num) 

class TdxhClipVison:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "bool_int": ("INT", {"default": 1, "min": 0, "max": 1, "step": 1}),

            "clip_name": (folder_paths.get_filename_list("clip_vision"), ), # CLIPVisionLoader

            # "clip_vision": ("CLIP_VISION",),
            "image": ("IMAGE",), # CLIPVisionEncode

            "conditioning": ("CONDITIONING", ),
            # "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
            "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "noise_augmentation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_adm"

    CATEGORY = "TDXH/tdxh_efficiency"

    def apply_adm(self,bool_int, clip_name, image, conditioning, strength, noise_augmentation):
        from nodes import CLIPVisionLoader, CLIPVisionEncode, unCLIPConditioning
        if bool_int == 0 or strength == 0:
            return (conditioning,)
        clip_vision = CLIPVisionLoader().load_clip(clip_name)[0]
        clip_vision_output = CLIPVisionEncode().encode(clip_vision,image)[0]
        return unCLIPConditioning().apply_adm(conditioning, clip_vision_output, strength, noise_augmentation)

if os.path.isdir(os.path.join(custom_nodes_dir, 'comfyui_controlnet_aux')):
    from custom_nodes.comfyui_controlnet_aux import AUX_NODE_MAPPINGS, AIO_NOT_SUPPORTED
else:
    AUX_NODE_MAPPINGS = {}
    AIO_NOT_SUPPORTED = []
from nodes import MAX_RESOLUTION
class TdxhControlNetProcessor:
    from nodes import ImageScale
    upscale_methods = ImageScale.upscale_methods
    crop_methods = ImageScale.crop_methods

    @classmethod
    def INPUT_TYPES(s):
        auxs = list(AUX_NODE_MAPPINGS.keys())
        for name in AIO_NOT_SUPPORTED:
            if name in auxs: auxs.remove(name)
        auxs.append("Invert")
        auxs.append("None")
        
        return {
            "required": { 
                "bool_int": ("INT", {"default": 1, "min": 0, "max": 1, "step": 1}),

                "image": ("IMAGE",), 
                "upscale_method": (s.upscale_methods,),
                "width": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "crop": (s.crop_methods,),

                # "image": ("IMAGE",),
                "preprocessor": (auxs, {"default": "None"})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "TDXH/tdxh_efficiency"

    def execute(self, bool_int,
                image, upscale_method, width, height, crop,
                preprocessor):
        from nodes import ImageScale, ImageInvert
        if bool_int == 0:
            return (image,)
        image = ImageScale().upscale(image, upscale_method, width, height, crop)[0]
        if preprocessor == "None":
            return (image,)
        if preprocessor == "Invert":
            return ImageInvert().invert(image)

        if os.path.isdir(os.path.join(custom_nodes_dir, 'comfyui_controlnet_aux')):
            from custom_nodes.comfyui_controlnet_aux import AIO_Preprocessor
            return AIO_Preprocessor().execute( preprocessor, image)
        else:
            return (image,)


class TdxhControlNetApply:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "bool_int": ("INT", {"default": 1, "min": 0, "max": 1, "step": 1}),

            "control_net_name": (folder_paths.get_filename_list("controlnet"), ),

            "positive": ("CONDITIONING", ),
            "negative": ("CONDITIONING", ),
            # "control_net": ("CONTROL_NET", ),
            "image": ("IMAGE", ),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
            }}

    RETURN_TYPES = ("CONDITIONING","CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_controlnet"

    CATEGORY = "TDXH/tdxh_efficiency"

    def apply_controlnet(self, bool_int, 
        control_net_name, 
        positive, negative, image, strength, start_percent, end_percent):
        from nodes import ControlNetLoader,ControlNetApplyAdvanced
        if bool_int == 0 or strength == 0:
            return (positive, negative)
        control_net=ControlNetLoader().load_controlnet(control_net_name)[0]
        return ControlNetApplyAdvanced().apply_controlnet(positive, negative, control_net, image, strength, start_percent, end_percent)



class TdxhReference:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "bool_int": ("INT", {"default": 1, "min": 0, "max": 1, "step": 1}),

            "main_latent":("LATENT",),

            "pixels": ("IMAGE", ), 
            "vae": ("VAE", ),

            "model": ("MODEL",),
            # "reference": ("LATENT",),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64})
            }}

    RETURN_TYPES = ("MODEL", "LATENT")
    FUNCTION = "reference_only"

    CATEGORY = "TDXH/tdxh_efficiency"

    def reference_only(self, bool_int, main_latent, pixels, vae, model, batch_size):
        if bool_int == 0:
            return (model,main_latent)
        from nodes import VAEEncode
        reference=VAEEncode().encode(vae, pixels)[0]
        if os.path.isdir(os.path.join(custom_nodes_dir, 'ComfyUI_experiments')):
            from custom_nodes.ComfyUI_experiments.reference_only import ReferenceOnlySimple
            return ReferenceOnlySimple().reference_only(model, reference, batch_size)
        elif os.path.isfile(os.path.join(custom_nodes_dir, 'reference_only.py')):
            from custom_nodes.reference_only import ReferenceOnlySimple
            return ReferenceOnlySimple().reference_only(model, reference, batch_size)
        else:
            raise RuntimeError("You must install custom_nodes: ComfyUI_experiments or provide reference_only.py!")

class TdxhImg2ImgLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "bool_int": ("INT", {"default": 1, "min": 0, "max": 1, "step": 1}),

            "main_latent":("LATENT",),
            "main_width": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
            "main_height": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),

            "pixels": ("IMAGE", ), 
            "vae": ("VAE", ),

            # "samples": ("LATENT",),
            "amount": ("INT", {"default": 1, "min": 1, "max": 64}),

            "pixels_width": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
            "pixels_height": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
            "denoise_img2img":("FLOAT", {"default": 0.5, "min": 0, "max": 1.0, "step": 0.05}),
            }}
    RETURN_TYPES = ("LATENT","INT","INT","FLOAT")
    RETURN_NAMES = ("LATENT","width_INT","height_INT","denoise")
    FUNCTION = "repeat"

    CATEGORY = "TDXH/tdxh_efficiency"

    def repeat(self, bool_int, main_latent, main_width,main_height, pixels, vae,  amount,pixels_width,pixels_height, denoise_img2img):
        if bool_int == 0:
            return (main_latent,main_width,main_height,1.0)
        from nodes import VAEEncode,RepeatLatentBatch
        samples = VAEEncode().encode(vae, pixels)[0]
        return (RepeatLatentBatch().repeat(samples,amount)[0],pixels_width,pixels_height, denoise_img2img)


class TdxhLtx23MultimodalDirector:
    DESCRIPTION = (
        "A multimodal control hub for LTX 2.3 style workflows. "
        "Combines text, image sequence, reference audio, and reference video frames "
        "into prompt blocks and a structured JSON control plan."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inputcount": ("INT", {"default": 4, "min": 1, "max": 24, "step": 1}),
                "text_prompt": ("STRING", {"multiline": True, "default": ""}),
                "subject_name": ("STRING", {"multiline": False, "default": ""}),
                "consistency_focus": (["character", "scene", "balanced"], {"default": "balanced"}),
                "image_reference_strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.5, "step": 0.05}),
                "audio_follow_mode": (["voice_and_scene", "voice_only", "scene_only"], {"default": "voice_and_scene"}),
                "audio_reference_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.5, "step": 0.05}),
                "video_follow_mode": (["style_camera_rhythm", "style_and_camera", "style_only"], {"default": "style_camera_rhythm"}),
                "video_reference_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.5, "step": 0.05}),
                "target_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "target_duration_seconds": ("FLOAT", {"default": 5.0, "min": 0.5, "max": 120.0, "step": 0.5}),
                "prompt_language": (["English", "Chinese", "Bilingual"], {"default": "English"}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "reference_audio": ("AUDIO",),
                "reference_video": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "INT", "FLOAT", "INT")
    RETURN_NAMES = (
        "ltx_prompt",
        "positive_prompt",
        "negative_prompt",
        "image_reference_guide",
        "audio_reference_guide",
        "video_reference_guide",
        "control_plan_json",
        "reference_image_count",
        "reference_audio_seconds",
        "reference_video_frames",
    )
    FUNCTION = "build"
    CATEGORY = "TDXH/tdxh_ltx23"

    def _language_hint(self, prompt_language):
        if prompt_language == "Chinese":
            return "Output prompt language: Chinese. Keep it concise, visual, and production-oriented."
        if prompt_language == "Bilingual":
            return "Output prompt language: bilingual Chinese and English, line by line."
        return "Output prompt language: English. Keep it concise, visual, and production-oriented."

    def _image_guide(self, subject_name, consistency_focus, image_count, image_frames, width, height, strength):
        subject_label = str(subject_name or "").strip() or "the same main subject"
        focus_map = {
            "character": "Prioritize face, hair, clothing, body proportion, and identity continuity.",
            "scene": "Prioritize environment layout, lighting atmosphere, color script, and prop continuity.",
            "balanced": "Balance subject identity continuity with environment continuity.",
        }
        return (
            f"Use the reference image sequence as the appearance anchor for {subject_label}. "
            f"Reference images connected: {image_count}, total image frames: {image_frames}, main size: {width}x{height}. "
            f"{focus_map[consistency_focus]} "
            f"Reference adherence strength: {strength:.2f}. "
            "Do not copy the frames mechanically; preserve recognizability while allowing natural motion and composition changes."
        )

    def _audio_guide(self, mode, duration_seconds, sample_rate, channels, strength):
        mode_map = {
            "voice_and_scene": "Follow both speaker timbre/emotion and the ambient scene sound cues.",
            "voice_only": "Follow the speaker timbre, pronunciation mood, and vocal texture; ignore unrelated ambient sound.",
            "scene_only": "Follow the ambience, acoustic space, and environmental sound texture; do not overfit a specific speaker identity.",
        }
        return (
            f"Use the reference audio as sound guidance. "
            f"Audio duration: {duration_seconds:.2f}s, sample rate: {sample_rate}, channels: {channels}. "
            f"{mode_map[mode]} "
            f"Reference adherence strength: {strength:.2f}. "
            "Keep the generated sound design coherent with the visual action and pacing."
        )

    def _video_guide(self, mode, frame_count, width, height, target_fps, duration_seconds, strength):
        mode_map = {
            "style_camera_rhythm": "Follow the reference video for overall style, camera language, and editing rhythm.",
            "style_and_camera": "Follow the reference video for style and shot design, but keep rhythm more flexible.",
            "style_only": "Use the reference video mainly as a style mood board instead of a strict motion template.",
        }
        return (
            f"Use the reference video frames as cinematography guidance. "
            f"Reference frames: {frame_count}, reference size: {width}x{height}. "
            f"{mode_map[mode]} "
            f"Target output: {duration_seconds:.2f}s at {target_fps:.2f} fps. "
            f"Reference adherence strength: {strength:.2f}. "
            "Extract lens feel, movement style, shot scale changes, and pacing, while keeping the requested content primary."
        )

    def _negative_prompt(self, consistency_focus, has_audio, has_video):
        negatives = [
            "low quality",
            "blurry",
            "flicker",
            "identity drift",
            "inconsistent face",
            "extra fingers",
            "deformed anatomy",
            "unstable composition",
            "overexposed",
            "underexposed",
            "oversaturated",
            "muddy details",
            "text artifacts",
            "watermark",
        ]
        if consistency_focus in ("scene", "balanced"):
            negatives.extend(["scene continuity errors", "prop continuity errors", "lighting continuity errors"])
        if has_audio:
            negatives.extend(["audio clipping", "mismatched lip sync", "noisy dialog", "disconnected ambience"])
        if has_video:
            negatives.extend(["jerky camera motion", "bad shot rhythm", "random zooming", "unmotivated cuts"])
        return ", ".join(negatives)

    def build(
        self,
        inputcount,
        text_prompt,
        subject_name,
        consistency_focus,
        image_reference_strength,
        audio_follow_mode,
        audio_reference_strength,
        video_follow_mode,
        video_reference_strength,
        target_fps,
        target_duration_seconds,
        prompt_language,
        image_1=None,
        reference_audio=None,
        reference_video=None,
        **kwargs,
    ):
        effective_images = _tdxh_collect_effective_images(inputcount, image_1, **kwargs)
        image_frames = 0
        image_width = 0
        image_height = 0
        for image in effective_images:
            frames, width, height = _tdxh_image_shape(image)
            image_frames += frames
            if width > 0 and height > 0 and image_width == 0 and image_height == 0:
                image_width, image_height = width, height

        video_frames, video_width, video_height = _tdxh_image_shape(reference_video)
        audio_seconds, audio_sample_rate, audio_channels = _tdxh_audio_stats(reference_audio)

        has_images = len(effective_images) > 0
        has_audio = audio_seconds > 0
        has_video = video_frames > 0

        positive_parts = [str(text_prompt or "").strip()]
        if not positive_parts[0]:
            positive_parts[0] = "Create a cinematic LTX 2.3 video shot."

        positive_parts.append(
            f"Target output: {target_duration_seconds:.2f}s, {target_fps:.2f} fps."
        )
        if str(subject_name or "").strip():
            positive_parts.append(f"Main subject: {subject_name.strip()}.")
        positive_parts.append(self._language_hint(prompt_language))

        image_reference_guide = ""
        if has_images:
            image_reference_guide = self._image_guide(
                subject_name,
                consistency_focus,
                len(effective_images),
                image_frames,
                image_width,
                image_height,
                image_reference_strength,
            )
            positive_parts.append(image_reference_guide)

        audio_reference_guide = ""
        if has_audio:
            audio_reference_guide = self._audio_guide(
                audio_follow_mode,
                audio_seconds,
                audio_sample_rate,
                audio_channels,
                audio_reference_strength,
            )
            positive_parts.append(audio_reference_guide)

        video_reference_guide = ""
        if has_video:
            video_reference_guide = self._video_guide(
                video_follow_mode,
                video_frames,
                video_width,
                video_height,
                target_fps,
                target_duration_seconds,
                video_reference_strength,
            )
            positive_parts.append(video_reference_guide)

        positive_prompt = "\n".join(part for part in positive_parts if part)
        negative_prompt = self._negative_prompt(consistency_focus, has_audio, has_video)

        control_plan = {
            "node": "TdxhLtx23MultimodalDirector",
            "text_prompt": str(text_prompt or "").strip(),
            "subject_name": str(subject_name or "").strip(),
            "consistency_focus": consistency_focus,
            "prompt_language": prompt_language,
            "target": {
                "duration_seconds": float(target_duration_seconds),
                "fps": float(target_fps),
            },
            "reference_summary": {
                "image_reference_count": len(effective_images),
                "image_reference_total_frames": image_frames,
                "image_reference_size": {"width": image_width, "height": image_height},
                "audio_reference_seconds": round(audio_seconds, 4),
                "audio_sample_rate": audio_sample_rate,
                "audio_channels": audio_channels,
                "video_reference_frames": video_frames,
                "video_reference_size": {"width": video_width, "height": video_height},
            },
            "weights": {
                "image_reference_strength": float(image_reference_strength),
                "audio_reference_strength": float(audio_reference_strength),
                "video_reference_strength": float(video_reference_strength),
            },
            "modes": {
                "audio_follow_mode": audio_follow_mode,
                "video_follow_mode": video_follow_mode,
            },
            "guides": {
                "image_reference_guide": image_reference_guide,
                "audio_reference_guide": audio_reference_guide,
                "video_reference_guide": video_reference_guide,
            },
            "negative_prompt": negative_prompt,
        }

        control_plan_json = json.dumps(control_plan, ensure_ascii=False, indent=2)
        ltx_prompt = f"{positive_prompt}\n\nNegative prompt:\n{negative_prompt}\n\nControl JSON:\n{control_plan_json}"
        return (
            ltx_prompt,
            positive_prompt,
            negative_prompt,
            image_reference_guide,
            audio_reference_guide,
            video_reference_guide,
            control_plan_json,
            len(effective_images),
            float(audio_seconds),
            video_frames,
        )


class TdxhLtx23RunnerConfig:
    DESCRIPTION = (
        "Bridge node for wiring TdxhLtx23MultimodalDirector outputs into a standard LTX 2.3 "
        "video generation chain. It computes frame count, trims reference audio if desired, "
        "selects the primary image anchor, and outputs LTX-friendly control values."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_prompt": ("STRING", {"forceInput": True}),
                "negative_prompt": ("STRING", {"forceInput": True}),
                "target_width": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 32}),
                "target_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32}),
                "target_fps": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "target_duration_seconds": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 120.0, "step": 0.5}),
                "audio_mode": (["trim_to_target", "keep_original", "disable_audio"], {"default": "trim_to_target"}),
                "use_reference_image_as_first_frame": ("BOOLEAN", {"default": True}),
                "use_vocals_only": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "reference_audio": ("AUDIO",),
                "reference_video": ("IMAGE",),
                "control_plan_json": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "IMAGE",
        "AUDIO",
        "IMAGE",
        "INT",
        "INT",
        "FLOAT",
        "INT",
        "FLOAT",
        "BOOLEAN",
        "BOOLEAN",
        "STRING",
    )
    RETURN_NAMES = (
        "positive_prompt",
        "negative_prompt",
        "primary_image",
        "audio_for_ltx",
        "reference_video",
        "width",
        "height",
        "fps",
        "total_frames",
        "audio_duration_seconds",
        "t2v_mode",
        "use_vocals_only",
        "runner_plan_json",
    )
    FUNCTION = "build"
    CATEGORY = "TDXH/tdxh_ltx23"

    def build(
        self,
        positive_prompt,
        negative_prompt,
        target_width,
        target_height,
        target_fps,
        target_duration_seconds,
        audio_mode,
        use_reference_image_as_first_frame,
        use_vocals_only,
        image_1=None,
        reference_audio=None,
        reference_video=None,
        control_plan_json="",
    ):
        target_width = int(target_width)
        target_height = int(target_height)
        target_fps = float(target_fps)
        target_duration_seconds = float(target_duration_seconds)

        total_frames = int(1 + 8 * round((target_duration_seconds * target_fps) / 8))
        primary_image = None if (not use_reference_image_as_first_frame or _tdxh_is_empty_image_input(image_1)) else image_1
        t2v_mode = primary_image is None

        audio_for_ltx = None
        if audio_mode == "keep_original":
            audio_for_ltx = reference_audio
        elif audio_mode == "trim_to_target":
            audio_for_ltx = _tdxh_trim_audio_to_seconds(reference_audio, target_duration_seconds)

        audio_duration_seconds, _audio_sample_rate, _audio_channels = _tdxh_audio_stats(audio_for_ltx)

        runner_plan = {
            "node": "TdxhLtx23RunnerConfig",
            "target": {
                "width": target_width,
                "height": target_height,
                "fps": target_fps,
                "duration_seconds": target_duration_seconds,
                "total_frames": total_frames,
            },
            "modes": {
                "audio_mode": audio_mode,
                "t2v_mode": bool(t2v_mode),
                "use_reference_image_as_first_frame": bool(use_reference_image_as_first_frame),
                "use_vocals_only": bool(use_vocals_only),
            },
            "reference_summary": {
                "has_primary_image": primary_image is not None,
                "has_audio": isinstance(audio_for_ltx, dict),
                "audio_duration_seconds": round(float(audio_duration_seconds), 4),
                "has_reference_video": not _tdxh_is_empty_image_input(reference_video),
            },
            "director_control_plan_json": str(control_plan_json or ""),
        }

        runner_plan_json = json.dumps(runner_plan, ensure_ascii=False, indent=2)

        return (
            str(positive_prompt or ""),
            str(negative_prompt or ""),
            primary_image,
            audio_for_ltx,
            reference_video,
            target_width,
            target_height,
            target_fps,
            total_frames,
            float(audio_duration_seconds),
            bool(t2v_mode),
            bool(use_vocals_only),
            runner_plan_json,
        )


class TdxhLtx23AllInOneBridge:
    DESCRIPTION = (
        "Single control hub for LTX 2.3 workflows. Accepts text plus batches of images, audios, "
        "and videos, then outputs the prompts and bridge values needed by the final LTX video chain."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_inputcount": ("INT", {"default": 4, "min": 1, "max": 24, "step": 1}),
                "audio_inputcount": ("INT", {"default": 1, "min": 1, "max": 12, "step": 1}),
                "video_inputcount": ("INT", {"default": 1, "min": 1, "max": 12, "step": 1}),
                "text_prompt": ("STRING", {"multiline": True, "default": ""}),
                "subject_name": ("STRING", {"multiline": False, "default": ""}),
                "consistency_focus": (["character", "scene", "balanced"], {"default": "balanced"}),
                "image_reference_strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.5, "step": 0.05}),
                "audio_follow_mode": (["voice_and_scene", "voice_only", "scene_only"], {"default": "voice_and_scene"}),
                "audio_reference_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.5, "step": 0.05}),
                "video_follow_mode": (["style_camera_rhythm", "style_and_camera", "style_only"], {"default": "style_camera_rhythm"}),
                "video_reference_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.5, "step": 0.05}),
                "target_width": ("INT", {"default": 1280, "min": 64, "max": 4096, "step": 32}),
                "target_height": ("INT", {"default": 736, "min": 64, "max": 4096, "step": 32}),
                "target_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "target_duration_seconds": ("FLOAT", {"default": 6.0, "min": 0.5, "max": 120.0, "step": 0.5}),
                "audio_mode": (["trim_to_target", "keep_original", "disable_audio"], {"default": "trim_to_target"}),
                "use_reference_image_as_first_frame": ("BOOLEAN", {"default": True}),
                "use_vocals_only": ("BOOLEAN", {"default": False}),
                "prompt_language": (["English", "Chinese", "Bilingual"], {"default": "English"}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "audio_1": ("AUDIO",),
                "video_1": ("IMAGE",),
            },
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
        "IMAGE",
        "AUDIO",
        "IMAGE",
        "INT",
        "INT",
        "FLOAT",
        "INT",
        "FLOAT",
        "BOOLEAN",
        "BOOLEAN",
        "STRING",
        "INT",
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "ltx_prompt",
        "positive_prompt",
        "negative_prompt",
        "primary_image",
        "audio_for_ltx",
        "primary_video_reference",
        "width",
        "height",
        "fps",
        "total_frames",
        "audio_duration_seconds",
        "t2v_mode",
        "use_vocals_only",
        "all_in_one_plan_json",
        "reference_image_count",
        "reference_audio_count",
        "reference_video_count",
    )
    FUNCTION = "build"
    CATEGORY = "TDXH/tdxh_ltx23"

    def build(
        self,
        image_inputcount,
        audio_inputcount,
        video_inputcount,
        text_prompt,
        subject_name,
        consistency_focus,
        image_reference_strength,
        audio_follow_mode,
        audio_reference_strength,
        video_follow_mode,
        video_reference_strength,
        target_width,
        target_height,
        target_fps,
        target_duration_seconds,
        audio_mode,
        use_reference_image_as_first_frame,
        use_vocals_only,
        prompt_language,
        image_1=None,
        audio_1=None,
        video_1=None,
        **kwargs,
    ):
        director = TdxhLtx23MultimodalDirector()

        effective_images = _tdxh_collect_effective_images(image_inputcount, image_1, **kwargs)
        effective_audios = _tdxh_collect_effective_media(
            audio_inputcount,
            "audio",
            audio_1,
            validator=_tdxh_is_valid_audio_input,
            **kwargs,
        )
        effective_videos = _tdxh_collect_effective_media(
            video_inputcount,
            "video",
            video_1,
            validator=lambda value: not _tdxh_is_empty_image_input(value),
            **kwargs,
        )

        primary_image_candidate = _tdxh_pick_first(effective_images)
        primary_audio_reference = _tdxh_pick_first(effective_audios)
        primary_video_reference = _tdxh_pick_first(effective_videos)

        image_frames = 0
        image_width = 0
        image_height = 0
        for image in effective_images:
            frames, width, height = _tdxh_image_shape(image)
            image_frames += frames
            if width > 0 and height > 0 and image_width == 0 and image_height == 0:
                image_width, image_height = width, height

        video_frames = 0
        video_width = 0
        video_height = 0
        for video in effective_videos:
            frames, width, height = _tdxh_image_shape(video)
            video_frames += frames
            if width > 0 and height > 0 and video_width == 0 and video_height == 0:
                video_width, video_height = width, height

        audio_total_seconds = 0.0
        primary_audio_seconds = 0.0
        primary_audio_sample_rate = 0
        primary_audio_channels = 0
        for idx, audio in enumerate(effective_audios):
            duration_seconds, sample_rate, channels = _tdxh_audio_stats(audio)
            audio_total_seconds += duration_seconds
            if idx == 0:
                primary_audio_seconds = duration_seconds
                primary_audio_sample_rate = sample_rate
                primary_audio_channels = channels

        has_images = len(effective_images) > 0
        has_audio = len(effective_audios) > 0
        has_video = len(effective_videos) > 0

        positive_parts = [str(text_prompt or "").strip()]
        if not positive_parts[0]:
            positive_parts[0] = "Create a cinematic LTX 2.3 video shot."

        positive_parts.append(
            f"Target output: {float(target_duration_seconds):.2f}s, {float(target_fps):.2f} fps, {int(target_width)}x{int(target_height)}."
        )
        if str(subject_name or "").strip():
            positive_parts.append(f"Main subject: {str(subject_name).strip()}.")
        positive_parts.append(director._language_hint(prompt_language))

        image_reference_guide = ""
        if has_images:
            image_reference_guide = director._image_guide(
                subject_name,
                consistency_focus,
                len(effective_images),
                image_frames,
                image_width,
                image_height,
                image_reference_strength,
            )
            positive_parts.append(image_reference_guide)

        audio_reference_guide = ""
        if has_audio:
            audio_reference_guide = director._audio_guide(
                audio_follow_mode,
                primary_audio_seconds,
                primary_audio_sample_rate,
                primary_audio_channels,
                audio_reference_strength,
            )
            if len(effective_audios) > 1:
                audio_reference_guide += f" Total audio references connected: {len(effective_audios)}, total duration: {audio_total_seconds:.2f}s."
            positive_parts.append(audio_reference_guide)

        video_reference_guide = ""
        if has_video:
            video_reference_guide = director._video_guide(
                video_follow_mode,
                video_frames,
                video_width,
                video_height,
                target_fps,
                target_duration_seconds,
                video_reference_strength,
            )
            if len(effective_videos) > 1:
                video_reference_guide += f" Total video references connected: {len(effective_videos)}."
            positive_parts.append(video_reference_guide)

        positive_prompt = "\n".join(part for part in positive_parts if part)
        negative_prompt = director._negative_prompt(consistency_focus, has_audio, has_video)
        total_frames = int(1 + 8 * round((float(target_duration_seconds) * float(target_fps)) / 8))

        primary_image = None
        if use_reference_image_as_first_frame and not _tdxh_is_empty_image_input(primary_image_candidate):
            primary_image = primary_image_candidate
        t2v_mode = primary_image is None

        audio_for_ltx = None
        if audio_mode == "keep_original":
            audio_for_ltx = primary_audio_reference
        elif audio_mode == "trim_to_target":
            audio_for_ltx = _tdxh_trim_audio_to_seconds(primary_audio_reference, target_duration_seconds)

        audio_duration_seconds, _audio_sample_rate, _audio_channels = _tdxh_audio_stats(audio_for_ltx)

        plan = {
            "node": "TdxhLtx23AllInOneBridge",
            "text_prompt": str(text_prompt or "").strip(),
            "subject_name": str(subject_name or "").strip(),
            "consistency_focus": consistency_focus,
            "prompt_language": prompt_language,
            "target": {
                "width": int(target_width),
                "height": int(target_height),
                "fps": float(target_fps),
                "duration_seconds": float(target_duration_seconds),
                "total_frames": total_frames,
            },
            "modes": {
                "audio_mode": audio_mode,
                "audio_follow_mode": audio_follow_mode,
                "video_follow_mode": video_follow_mode,
                "t2v_mode": bool(t2v_mode),
                "use_reference_image_as_first_frame": bool(use_reference_image_as_first_frame),
                "use_vocals_only": bool(use_vocals_only),
            },
            "weights": {
                "image_reference_strength": float(image_reference_strength),
                "audio_reference_strength": float(audio_reference_strength),
                "video_reference_strength": float(video_reference_strength),
            },
            "reference_summary": {
                "image_reference_count": len(effective_images),
                "image_reference_total_frames": image_frames,
                "image_reference_size": {"width": image_width, "height": image_height},
                "audio_reference_count": len(effective_audios),
                "audio_reference_total_seconds": round(float(audio_total_seconds), 4),
                "primary_audio_seconds": round(float(primary_audio_seconds), 4),
                "audio_for_ltx_seconds": round(float(audio_duration_seconds), 4),
                "video_reference_count": len(effective_videos),
                "video_reference_total_frames": video_frames,
                "video_reference_size": {"width": video_width, "height": video_height},
            },
            "guides": {
                "image_reference_guide": image_reference_guide,
                "audio_reference_guide": audio_reference_guide,
                "video_reference_guide": video_reference_guide,
            },
            "negative_prompt": negative_prompt,
        }

        all_in_one_plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
        ltx_prompt = f"{positive_prompt}\n\nNegative prompt:\n{negative_prompt}\n\nControl JSON:\n{all_in_one_plan_json}"
        return (
            ltx_prompt,
            positive_prompt,
            negative_prompt,
            primary_image,
            audio_for_ltx,
            primary_video_reference,
            int(target_width),
            int(target_height),
            float(target_fps),
            total_frames,
            float(audio_duration_seconds),
            bool(t2v_mode),
            bool(use_vocals_only),
            all_in_one_plan_json,
            len(effective_images),
            len(effective_audios),
            len(effective_videos),
        )


class TdxhLtx23MultimodalVideoGenerator:
    DESCRIPTION = (
        "End-to-end LTX 2.3 generation node. Accepts text plus batches of images, audios, and videos, "
        "then produces decoded video frames, optional audio, and fps for the final Save Video node."
    )

    @classmethod
    def INPUT_TYPES(cls):
        from nodes import VAELoader

        return {
            "required": {
                "image_inputcount": ("INT", {"default": 4, "min": 1, "max": 24, "step": 1}),
                "audio_inputcount": ("INT", {"default": 1, "min": 1, "max": 12, "step": 1}),
                "video_inputcount": ("INT", {"default": 1, "min": 1, "max": 12, "step": 1}),
                "text_prompt": ("STRING", {"multiline": True, "default": ""}),
                "subject_name": ("STRING", {"multiline": False, "default": ""}),
                "consistency_focus": (["character", "scene", "balanced"], {"default": "balanced"}),
                "image_reference_strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.5, "step": 0.05}),
                "audio_follow_mode": (["voice_and_scene", "voice_only", "scene_only"], {"default": "voice_and_scene"}),
                "audio_reference_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.5, "step": 0.05}),
                "video_follow_mode": (["style_camera_rhythm", "style_and_camera", "style_only"], {"default": "style_camera_rhythm"}),
                "video_reference_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.5, "step": 0.05}),
                "target_width": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 32}),
                "target_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32}),
                "target_fps": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "target_duration_seconds": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 120.0, "step": 0.5}),
                "audio_mode": (["trim_to_target", "keep_original", "disable_audio"], {"default": "trim_to_target"}),
                "use_reference_image_as_first_frame": ("BOOLEAN", {"default": True}),
                "use_vocals_only": ("BOOLEAN", {"default": False}),
                "prompt_language": (["English", "Chinese", "Bilingual"], {"default": "English"}),
                "unet_name": (folder_paths.get_filename_list("diffusion_models"), {"default": "ltx-2.3-22b-distilled_transformer_only_fp8_input_scaled_v3.safetensors"}),
                "text_encoder_gemma": (folder_paths.get_filename_list("text_encoders"), {"default": "gemma_3_12B_it_fp8_scaled.safetensors"}),
                "text_encoder_projection": (folder_paths.get_filename_list("text_encoders"), {"default": "ltx-2.3_text_projection_bf16.safetensors"}),
                "video_vae_name": (VAELoader.vae_list(VAELoader), {"default": "LTX23_video_vae_bf16.safetensors"}),
                "audio_vae_name": (TdxhVAELoader.vae_list(), {"default": "LTX23_audio_vae_bf16-KJ.safetensors"}),
                "latent_upscale_model_name": (folder_paths.get_filename_list("latent_upscale_models"), {"default": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"}),
                "first_pass_seed": ("INT", {"default": 43, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "first_pass_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "first_pass_steps": ("INT", {"default": 8, "min": 1, "max": 100, "step": 1}),
                "first_pass_sampler": (["euler_ancestral_cfg_pp", "euler", "euler_cfg_pp"], {"default": "euler_ancestral_cfg_pp"}),
                "second_pass_seed": ("INT", {"default": 420, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "second_pass_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "second_pass_sampler": (["euler_cfg_pp", "euler", "euler_ancestral_cfg_pp"], {"default": "euler_cfg_pp"}),
                "second_pass_sigmas": ("STRING", {"default": "0.85, 0.7250, 0.4219, 0.0", "multiline": False}),
                "image_preprocess_compression": ("INT", {"default": 33, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "audio_1": ("AUDIO",),
                "video_1": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("images", "audio", "fps", "ltx_prompt", "generation_plan_json")
    FUNCTION = "generate"
    CATEGORY = "TDXH/tdxh_ltx23"

    def generate(
        self,
        image_inputcount,
        audio_inputcount,
        video_inputcount,
        text_prompt,
        subject_name,
        consistency_focus,
        image_reference_strength,
        audio_follow_mode,
        audio_reference_strength,
        video_follow_mode,
        video_reference_strength,
        target_width,
        target_height,
        target_fps,
        target_duration_seconds,
        audio_mode,
        use_reference_image_as_first_frame,
        use_vocals_only,
        prompt_language,
        unet_name,
        text_encoder_gemma,
        text_encoder_projection,
        video_vae_name,
        audio_vae_name,
        latent_upscale_model_name,
        first_pass_seed,
        first_pass_cfg,
        first_pass_steps,
        first_pass_sampler,
        second_pass_seed,
        second_pass_cfg,
        second_pass_sampler,
        second_pass_sigmas,
        image_preprocess_compression,
        image_1=None,
        audio_1=None,
        video_1=None,
        **kwargs,
    ):
        from nodes import CLIPTextEncode, DualCLIPLoader, UNETLoader, VAEDecodeTiled, VAELoader
        from comfy_extras.nodes_custom_sampler import CFGGuider, KSamplerSelect, ManualSigmas, RandomNoise, SamplerCustomAdvanced
        from comfy_extras.nodes_hunyuan import LatentUpscaleModelLoader
        from comfy_extras.nodes_lt import EmptyLTXVLatentVideo, LTXVConditioning, LTXVConcatAVLatent, LTXVImgToVideoInplace, LTXVScheduler, LTXVSeparateAVLatent, LTXVPreprocess
        from comfy_extras.nodes_lt_audio import LTXVAudioVAEEncode, LTXVEmptyLatentAudio
        from comfy_extras.nodes_lt_upsampler import LTXVLatentUpsampler

        bridge_outputs = TdxhLtx23AllInOneBridge().build(
            image_inputcount=image_inputcount,
            audio_inputcount=audio_inputcount,
            video_inputcount=video_inputcount,
            text_prompt=text_prompt,
            subject_name=subject_name,
            consistency_focus=consistency_focus,
            image_reference_strength=image_reference_strength,
            audio_follow_mode=audio_follow_mode,
            audio_reference_strength=audio_reference_strength,
            video_follow_mode=video_follow_mode,
            video_reference_strength=video_reference_strength,
            target_width=target_width,
            target_height=target_height,
            target_fps=target_fps,
            target_duration_seconds=target_duration_seconds,
            audio_mode=audio_mode,
            use_reference_image_as_first_frame=use_reference_image_as_first_frame,
            use_vocals_only=use_vocals_only,
            prompt_language=prompt_language,
            image_1=image_1,
            audio_1=audio_1,
            video_1=video_1,
            **kwargs,
        )

        (
            ltx_prompt,
            positive_prompt,
            negative_prompt,
            primary_image,
            audio_for_ltx,
            _primary_video_reference,
            width,
            height,
            fps,
            total_frames,
            _audio_duration_seconds,
            t2v_mode,
            _use_vocals_only,
            bridge_plan_json,
            _reference_image_count,
            _reference_audio_count,
            _reference_video_count,
        ) = bridge_outputs

        model = UNETLoader().load_unet(unet_name, "default")[0]
        clip = DualCLIPLoader().load_clip(text_encoder_gemma, text_encoder_projection, "ltxv", "default")[0]
        video_vae = VAELoader().load_vae(video_vae_name)[0]
        audio_vae = TdxhVAELoader().load_vae(audio_vae_name, "main_device", "bf16")[0]
        upscale_model = LatentUpscaleModelLoader.execute(latent_upscale_model_name)[0]

        positive = CLIPTextEncode().encode(clip, positive_prompt)[0]
        negative = CLIPTextEncode().encode(clip, negative_prompt)[0]
        positive, negative = LTXVConditioning.execute(positive, negative, fps)

        video_latent = EmptyLTXVLatentVideo.execute(width, height, total_frames, 1)[0]

        if isinstance(audio_for_ltx, dict):
            audio_latent = LTXVAudioVAEEncode.execute(audio_for_ltx, audio_vae)[0]
            final_audio = audio_for_ltx
        else:
            audio_latent = LTXVEmptyLatentAudio.execute(total_frames, max(int(round(fps)), 1), 1, audio_vae)[0]
            final_audio = None

        if (not t2v_mode) and (primary_image is not None):
            preprocessed_image = LTXVPreprocess.execute(primary_image, int(image_preprocess_compression))[0]
            video_latent = LTXVImgToVideoInplace.execute(video_vae, preprocessed_image, video_latent, 1.0, False)[0]
        else:
            preprocessed_image = None

        first_pass_latent = LTXVConcatAVLatent.execute(video_latent, audio_latent)[0]
        first_noise = RandomNoise.execute(int(first_pass_seed))[0]
        first_guider = CFGGuider.execute(model, positive, negative, float(first_pass_cfg))[0]
        first_sampler = KSamplerSelect.execute(first_pass_sampler)[0]
        first_sigmas = LTXVScheduler.execute(int(first_pass_steps), 2.05, 0.95, True, 0.1, first_pass_latent)[0]
        first_pass_latent = SamplerCustomAdvanced.execute(first_noise, first_guider, first_sampler, first_sigmas, first_pass_latent)[0]

        first_video_latent, first_audio_latent = LTXVSeparateAVLatent.execute(first_pass_latent)
        upscaled_video_latent = LTXVLatentUpsampler().upsample_latent(first_video_latent, upscale_model, video_vae)[0]

        if preprocessed_image is not None:
            upscaled_video_latent = LTXVImgToVideoInplace.execute(video_vae, preprocessed_image, upscaled_video_latent, 1.0, False)[0]

        second_pass_latent = LTXVConcatAVLatent.execute(upscaled_video_latent, first_audio_latent)[0]
        second_noise = RandomNoise.execute(int(second_pass_seed))[0]
        second_guider = CFGGuider.execute(model, positive, negative, float(second_pass_cfg))[0]
        second_sampler_obj = KSamplerSelect.execute(second_pass_sampler)[0]
        second_sigmas_obj = ManualSigmas.execute(second_pass_sigmas)[0]
        second_pass_latent = SamplerCustomAdvanced.execute(second_noise, second_guider, second_sampler_obj, second_sigmas_obj, second_pass_latent)[1]
        final_video_latent, _final_audio_latent = LTXVSeparateAVLatent.execute(second_pass_latent)
        decoded_images = VAEDecodeTiled().decode(video_vae, final_video_latent, 512, 64, 4096, 8)[0]

        generation_plan = json.loads(bridge_plan_json)
        generation_plan["node"] = "TdxhLtx23MultimodalVideoGenerator"
        generation_plan["generation"] = {
            "unet_name": unet_name,
            "text_encoder_gemma": text_encoder_gemma,
            "text_encoder_projection": text_encoder_projection,
            "video_vae_name": video_vae_name,
            "audio_vae_name": audio_vae_name,
            "latent_upscale_model_name": latent_upscale_model_name,
            "first_pass": {
                "seed": int(first_pass_seed),
                "cfg": float(first_pass_cfg),
                "steps": int(first_pass_steps),
                "sampler": first_pass_sampler,
            },
            "second_pass": {
                "seed": int(second_pass_seed),
                "cfg": float(second_pass_cfg),
                "sampler": second_pass_sampler,
                "sigmas": str(second_pass_sigmas),
            },
            "image_preprocess_compression": int(image_preprocess_compression),
        }
        generation_plan_json = json.dumps(generation_plan, ensure_ascii=False, indent=2)

        return (
            decoded_images,
            final_audio,
            float(fps),
            str(ltx_prompt or ""),
            generation_plan_json,
        )


NODE_CLASS_MAPPINGS = {
    # tdxh_image
    "TdxhImageToSize": TdxhImageToSize,
    "TdxhImageToSizeAdvanced":TdxhImageToSizeAdvanced,
    # tdxh_model
    "TdxhLoraLoader":TdxhLoraLoader,
    "TdxhVAELoader":TdxhVAELoader,
    # tdxh_data
    "TdxhIntInput":TdxhIntInput,
    "TdxhFloatInput":TdxhFloatInput,
    "TdxhStringInput":TdxhStringInput,
    "TdxhSaveText":TdxhSaveText,
    "TdxhStringInputTranslator":TdxhStringInputTranslator,
    # tdxh_bool
    "TdxhOnOrOff":TdxhOnOrOff,
    "TdxhBoolNumber":TdxhBoolNumber,
    "TdxhToggleMaster":TdxhToggleMaster,
    "TdxhToggleGuest":TdxhToggleGuest,
    # tdxh_efficiency
    "TdxhClipVison" : TdxhClipVison,
    "TdxhControlNetProcessor":TdxhControlNetProcessor,
    "TdxhControlNetApply":TdxhControlNetApply,
    "TdxhReference":TdxhReference,
    "TdxhImg2ImgLatent":TdxhImg2ImgLatent,
    # tdxh_ltx23
    "TdxhLtx23MultimodalDirector":TdxhLtx23MultimodalDirector,
    "TdxhLtx23RunnerConfig":TdxhLtx23RunnerConfig,
    "TdxhLtx23AllInOneBridge":TdxhLtx23AllInOneBridge,
    "TdxhLtx23MultimodalVideoGenerator":TdxhLtx23MultimodalVideoGenerator,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    # tdxh_image
    "TdxhImageToSize": "TdxhImageToSize",
    "TdxhImageToSizeAdvanced":"TdxhImageToSizeAdvanced",
    # tdxh_model
    "TdxhLoraLoader":"TdxhLoraLoader",
    "TdxhVAELoader":"TdxhVAELoader",
    # tdxh_data
    "TdxhIntInput":"TdxhIntInput",
    "TdxhFloatInput":"TdxhFloatInput",
    "TdxhStringInput":"TdxhStringInput",
    "TdxhSaveText":"TdxhSaveText",
    "TdxhStringInputTranslator":"TdxhStringInputTranslator",
    # tdxh_bool
    "TdxhOnOrOff":"TdxhOnOrOff",
    "TdxhBoolNumber":"TdxhBoolNumber",
    "TdxhToggleMaster":"TdxhToggleMaster",
    "TdxhToggleGuest":"TdxhToggleGuest",
    # tdxh_efficiency
    "TdxhClipVison" : "TdxhClipVison",
    "TdxhControlNetProcessor":"TdxhControlNetProcessor",
    "TdxhControlNetApply":"TdxhControlNetApply",
    "TdxhReference":"TdxhReference",
    "TdxhImg2ImgLatent":"TdxhImg2ImgLatent",
    # tdxh_ltx23
    "TdxhLtx23MultimodalDirector":"TdxhLtx23MultimodalDirector",
    "TdxhLtx23RunnerConfig":"TdxhLtx23RunnerConfig",
    "TdxhLtx23AllInOneBridge":"TdxhLtx23AllInOneBridge",
    "TdxhLtx23MultimodalVideoGenerator":"TdxhLtx23MultimodalVideoGenerator",

}




