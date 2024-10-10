from PIL import Image
import numpy as np
import os
import sys

# sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import comfy.utils
import comfy.sd

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
        else:
            if bool_int_from_master==1:
                bool_num =bool_int
            else:
                bool_num = 0
        return (bool_num, bool_num)

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
        from custom_nodes.reference_only import ReferenceOnlySimple
        reference=VAEEncode().encode(vae, pixels)[0]
        return ReferenceOnlySimple().reference_only(model, reference, batch_size)

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
    # tdxh_data
    "TdxhIntInput":TdxhIntInput,
    "TdxhFloatInput":TdxhFloatInput,
    "TdxhStringInput":TdxhStringInput,
    "TdxhStringInputTranslator":TdxhStringInputTranslator,
    # tdxh_bool
    "TdxhOnOrOff":TdxhOnOrOff,
    "TdxhBoolNumber":TdxhBoolNumber,
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
    # tdxh_data
    "TdxhIntInput":"TdxhIntInput",
    "TdxhFloatInput":"TdxhFloatInput",
    "TdxhStringInput":"TdxhStringInput",
    "TdxhStringInputTranslator":"TdxhStringInputTranslator",
    # tdxh_bool
    "TdxhOnOrOff":"TdxhOnOrOff",
    "TdxhBoolNumber":"TdxhBoolNumber",
    # tdxh_efficiency
    "TdxhClipVison" : "TdxhClipVison",
    "TdxhControlNetProcessor":"TdxhControlNetProcessor",
    "TdxhControlNetApply":"TdxhControlNetApply",
    "TdxhReference":"TdxhReference",
    "TdxhImg2ImgLatent":"TdxhImg2ImgLatent",

}




