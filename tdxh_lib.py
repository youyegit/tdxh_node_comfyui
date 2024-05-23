SD_XL_BASE_RATIOS = {
    "0.25":(512, 2048), # new
    "0.26":(512, 1984), # new
    "0.27":(512, 1920), # new
    "0.28":(512, 1856), # new
    "0.32":(576, 1792), # new
    "0.33": (576, 1728), # new
    "0.35": (576, 1664), # new
    "0.4":(640, 1600), # new
    "0.42": (640, 1536), # new
    "0.48":(704, 1472), # new
    "0.5": (704, 1408),
    "0.52": (704, 1344),
    "0.57": (768, 1344),
    "0.6": (768, 1280),
    "0.68": (832, 1216),
    "0.72": (832, 1152),
    "0.78": (896, 1152),
    "0.82": (896, 1088),
    "0.88": (960, 1088),
    "0.94": (960, 1024),
    "1.0": (1024, 1024),
    "1.07": (1024, 960),
    "1.13": (1088, 960),
    "1.21": (1088, 896),
    "1.29": (1152, 896),
    "1.38": (1152, 832),
    "1.46": (1216, 832),
    "1.67": (1280, 768),
    "1.75": (1344, 768),
    "1.91": (1344, 704), # not in training value but in Stability-AI/generative-models/scripts/demo/sampling.py
    "2.0": (1408, 704),
    "2.09": (1472, 704),
    "2.4": (1536, 640),
    "2.5": (1600, 640),
    "2.89": (1664, 576),
    "3.0": (1728, 576),
    "3.11":(1792, 576), # new
    "3.62":(1856, 512), # new
    "3.75":(1920, 512), # new
    "3.88":(1984, 512), # new
    "4.0":(2048, 512), # new
    }

target_sizes_show = [f"{k}:{v}" for k, v in SD_XL_BASE_RATIOS.items()]


def get_SDXL_best_size(image_size = None, ratio = None):
    """
    input a tuple such as get_SDXL_best_size((1200, 900)or input a float num such as get_SDXL_best_size(ratio=1.3)
    return a tuple for SDXL such as (1152, 832)
    """
    best_size = None
    if image_size:
        if image_size[0] > 0 and image_size[1] > 0:
            ratio = image_size[0] / image_size[1] # w, h = image_size
    if ratio:
        target_sizes = [v for _, v in SD_XL_BASE_RATIOS.items()]
        min_diff = float('inf') # a variable to store the minimum difference
        for target_size in target_sizes:
            target_ratio = target_size[0] / target_size[1]
            diff = abs(ratio - target_ratio)
            if diff < min_diff:
                min_diff = diff
                best_size = target_size
    return best_size

# Test the function
def test1():
    print(get_SDXL_best_size((800, 800))) # Output (1024, 1024)
    print(get_SDXL_best_size((1200, 900))) # Output (1152, 896)
    print(get_SDXL_best_size((600, 800))) # Output (896, 1152) # (832, 1152)
    print(get_SDXL_best_size((700, 500))) # Output (1216, 832) # (1152, 832)
    print(get_SDXL_best_size((1080, 1920))) # Output (768, 1344)
    print(get_SDXL_best_size((1200, 500))) # Output (1536, 640)
    print(get_SDXL_best_size(ratio=1.3)) # Output (1152, 896)
    print(get_SDXL_best_size(ratio=1)) # Output (1024, 1024)

# test1()
# print(target_sizes_show)