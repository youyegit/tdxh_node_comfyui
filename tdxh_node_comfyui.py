from PIL import Image
import numpy as np
# import os
# import sys

# sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import comfy.utils
import comfy.sd

import folder_paths

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
                "ratio": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 10.0, 
                    "step": 0.1
                }),
                "what_to_follow": ([
                    "only_width", "only_height", "both_width_and_height", "only_ratio", 
                    "only_image"
                ],),
            }
        }

    RETURN_TYPES = ("INT","INT","FLOAT","FLOAT","STRING","STRING","NUMBER","NUMBER")
    RETURN_NAMES = ("width_INT", "height_INT","width_FLOAT", "height_FLOAT","width_STRING", "height_STRING","width_NUMBER", "height_NUMBER")
    FUNCTION = "tdxh_image_to_size_advanced"
    #OUTPUT_NODE = False
    CATEGORY = "TDXH/tdxh_image"

    def tdxh_image_to_size_advanced(self, image, width, height, ratio,what_to_follow):
        image_size = self.tdxh_image_to_size(image)
        # width = self.tdxh_nearest_divisible_by_8(width)
        # height = self.tdxh_nearest_divisible_by_8(height)
        # w, h = image_size[0], image_size[1]
        if what_to_follow == "only_image":
            return image_size
        elif what_to_follow == "only_ratio":
            w, h = ratio * image_size[0], ratio * image_size[1]
            w, h = self.tdxh_nearest_divisible_by_8(w), self.tdxh_nearest_divisible_by_8(h)
        elif what_to_follow == "both_width_and_height":
            w, h = width, height
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
            "model": ("MODEL",),
            "clip": ("CLIP", ),
            "enable_or_not": ([
                "enable", 
                "disable"
                ],),
            "lora_name": (folder_paths.get_filename_list("loras"), ),
            "strength_both": ("FLOAT", {
                "default": 0.5, "min": -10.0, 
                "max": 10.0, "step": 0.05
                }),
            "strength_model": ("FLOAT", {
                "default": 0.5, "min": -10.0, 
                "max": 10.0, "step": 0.05
                }),
            "strength_clip": ("FLOAT", {
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

    def load_lora(self, model, clip, enable_or_not, lora_name, strength_both,strength_model, strength_clip, what_to_follow):
        if enable_or_not == "disable":
            return (model, clip)
        if what_to_follow == "only_strength_both":
            strength_model, strength_clip = strength_both, strength_both

        if strength_model == 0 and strength_clip == 0:
            return (model, clip)
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora)

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
}




