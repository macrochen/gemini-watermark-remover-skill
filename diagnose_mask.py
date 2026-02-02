import os
from PIL import Image
import numpy as np

mask_path = os.path.expanduser("~/.gemini/assets/masks/bg_96.png")
try:
    img = Image.open(mask_path)
    print(f"Format: {img.format}")
    print(f"Mode: {img.mode}")
    print(f"Size: {img.size}")
    
    if img.mode == 'RGBA':
        r, g, b, a = img.split()
        print(f"Alpha channel range: {a.getextrema()}")
        print(f"Red channel range: {r.getextrema()}")
        # Check if RGB is uniform
        np_r = np.array(r)
        print(f"Mean Red: {np_r.mean()}")
    elif img.mode == 'L':
        print(f"L channel range: {img.getextrema()}")
    
except Exception as e:
    print(f"Error: {e}")
