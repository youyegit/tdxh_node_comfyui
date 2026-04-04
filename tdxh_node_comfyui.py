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

}




