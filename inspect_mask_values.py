import os
from PIL import Image
import numpy as np

mask_path = os.path.expanduser("~/.gemini/assets/masks/bg_96.png")
img = Image.open(mask_path)
arr = np.array(img)
print(f"Min: {arr.min()}, Max: {arr.max()}, Mean: {arr.mean()}")
# Print center pixel
h, w, c = arr.shape
print(f"Center pixel: {arr[h//2, w//2]}")
# Print corner pixel
print(f"Corner pixel (0,0): {arr[0,0]}")
