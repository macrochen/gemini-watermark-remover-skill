import os
import numpy as np
from PIL import Image

mask_path = os.path.expanduser("~/.gemini/assets/masks/bg_96.png")
image_path = os.path.expanduser("~/Downloads/Gemini_Generated_Image_ag6t1rag6t1rag6t.png")

mask = Image.open(mask_path).convert("RGB")
img = Image.open(image_path).convert("RGB")

mask_arr = np.array(mask).astype(float)
img_arr = np.array(img).astype(float)

mh, mw, _ = mask_arr.shape
ih, iw, _ = img_arr.shape

# 我们在右下角 150x150 的范围内寻找最佳匹配点
search_area_size = 150
search_y_start = ih - search_area_size
search_x_start = iw - search_area_size

best_pos = (0, 0)
min_diff = float('inf')

print(f"Scanning for watermark in {search_area_size}x{search_area_size} area...")

# 简单的模板匹配：寻找减去水印后，该区域像素平均亮度最低（即最接近原始底色）的位置
for y in range(search_y_start, ih - mh):
    for x in range(search_x_start, iw - mw):
        patch = img_arr[y:y+mh, x:x+mw]
        # 模拟减法去除后的残余（简单模型）
        diff = np.mean(np.abs(patch - mask_arr))
        if diff < min_diff:
            min_diff = diff
            best_pos = (x, y)

offset_x = iw - mw - best_pos[0]
offset_y = ih - mh - best_pos[1]

print(f"Best match found at: {best_pos}")
print(f"Offset from bottom-right: dx={offset_x}, dy={offset_y}")
