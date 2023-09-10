
# Define a function that takes a tuple representing the image width and height as a parameter
def get_SDXL_best_size(image_size):
    # Assign the image width and height to w and h respectively
    w, h = image_size
    # Calculate the image aspect ratio
    ratio = w / h
    # Define a list to store the target sizes

    SD_XL_BASE_RATIOS = {
    "0.33": (576, 1728), # 推测新增
    "0.35": (576, 1664), # 推测新增
    "0.42": (640, 1536), # 新增
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
    "1.91": (1344, 704),
    "2.0": (1408, 704),
    "2.09": (1472, 704),
    "2.4": (1536, 640),
    "2.5": (1600, 640),
    "2.89": (1664, 576),
    "3.0": (1728, 576),
    }

    target_sizes_more = [v for _, v in SD_XL_BASE_RATIOS.items()]
    # print(target_sizes_more)
    target_sizes = target_sizes_more

    # target_sizes = [
    #     (1024, 1024), # 1         # 1/1
    #     (1152, 896), # 1.2857...  # 4/3
    #     (896, 1152), # 0.7777...  # 3/4
    #     (1216, 832), # 1.4615...  # 3/2 7/5
    #     (832, 1216), # 0.6842...  # 2/3 5/7
    #     (1344, 768), # 1.75       # 16/9
    #     (768, 1344), # 0.5714...  # 9/16
    #     (1536, 640), # 2.4        # 12/5
    #     (640, 1536)  # 0.4166     # 5/12
    # ]


    # Define a variable to store the minimum difference
    min_diff = float('inf')
    # Define a variable to store the closest target size
    best_size = None
    # Loop through the target size list
    for target_size in target_sizes:
        # Calculate the target size aspect ratio
        target_ratio = target_size[0] / target_size[1]
        # Calculate the absolute value of the difference between the image aspect ratio and the target size aspect ratio
        diff = abs(ratio - target_ratio)
        # If the difference is smaller than the minimum difference
        if diff < min_diff:
            # Update the minimum difference and the closest target size
            min_diff = diff
            best_size = target_size
            # Return the closest target size
    return best_size

# Test the function
def test1():
    print(get_SDXL_best_size((800, 800))) # Output (1024, 1024)
    print(get_SDXL_best_size((1200, 900))) # Output (1152, 896)
    print(get_SDXL_best_size((600, 800))) # Output (896, 1152) # (832, 1152)
    print(get_SDXL_best_size((700, 500))) # Output (1216, 832) # (1152, 832)
    print(get_SDXL_best_size((1080, 1920))) # Output (768, 1344)
    print(get_SDXL_best_size((1200, 500))) # Output (1536, 640)

# test1()
