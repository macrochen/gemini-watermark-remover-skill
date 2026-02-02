import os
import numpy as np
from PIL import Image

mask_path = os.path.expanduser("~/.gemini/assets/masks/bg_96.png")
image_path = os.path.expanduser("~/Downloads/Gemini_Generated_Image_ag6t1rag6t1rag6t.png")

try:
    mask = Image.open(mask_path).convert("RGB")
    img = Image.open(image_path).convert("RGB")
    
    mask_arr = np.array(mask)
    img_arr = np.array(img)
    
    mh, mw, _ = mask_arr.shape
    ih, iw, _ = img_arr.shape
    
    # Check mask corners
    print(f"Mask Size: {mw}x{mh}")
    print(f"Mask Top-Left: {mask_arr[0,0]}")
    print(f"Mask Bottom-Right: {mask_arr[-1,-1]}")
    
    # Template Matching to find exact position
    # We only search in the bottom-right area to save time
    search_margin = 50 # pixels to search around the bottom-right corner
    
    # Extract candidate region from image (bottom-right corner + some buffer)
    # We assume the mask is somewhere in the bottom-right 200x200 area
    search_h = 200
    search_w = 200
    
    if ih < search_h or iw < search_w:
        print("Image too small for search.")
        exit()
        
    search_region = img_arr[ih-search_h:, iw-search_w:]
    
    # We want to find where 'mask' fits best in 'search_region'.
    # Since we are looking for a transparent overlay, standard SSD (Sum of Squared Differences) might work 
    # if we assume the watermark is additive. 
    # But simpler: The mask is bright where the watermark is. The image is also bright-ish there.
    # Let's just do a brute force search for the position that minimizes the variance of the recovered pixels?
    # Or simply finding the spot where the image is brightest matching the mask?
    
    # Actually, let's look at pixel difference.
    # Since alpha blending is O = (I - A*L)/(1-A)
    # If aligned correctly, O should be "smooth" or "natural".
    # If misaligned, O will have artifacts (like black pixels).
    
    best_offset = (0, 0)
    min_artifact_score = float('inf')
    
    # Convert mask to alpha map for calculation
    alpha_map = np.max(mask_arr, axis=2) / 255.0
    
    # Optimization: Only check every pixel or every other pixel? 
    # Let's search for the mask's bottom-right corner relative to image's bottom-right corner.
    # Offset (0,0) means mask's bottom-right aligns with image's bottom-right.
    # Offset (dx, dy) means we shift mask left by dx, up by dy.
    
    print("Searching for best alignment...")
    
    for dy in range(0, 40): # Search up to 40px up
        for dx in range(0, 40): # Search up to 40px left
            
            # Coordinates in Image
            # Mask Top-Left at:
            # y = ih - mh - dy
            # x = iw - mw - dx
            
            y = ih - mh - dy
            x = iw - mw - dx
            
            if y < 0 or x < 0: continue
            
            # Extract patch from image
            patch = img_arr[y:y+mh, x:x+mw].astype(float)
            
            # Calculate restored patch
            # score = sum of absolute gradients? or just count of black pixels?
            # When misaligned, we get black pixels (value < 0 clipped to 0).
            # So let's count how many pixels become < 0 before clipping.
            
            # Simplified restoration for checking:
            # val = patch - alpha * 255
            # We don't need to divide by (1-alpha) to check for negativity.
            # If patch < alpha * 255, it's a "black artifact".
            
            # Broadcast alpha
            alpha_exp = alpha_map[:, :, np.newaxis]
            
            diff = patch - alpha_exp * 255.0
            
            # Count pixels that are significantly negative
            # (Allowing some noise)
            negative_pixels = np.sum(diff < -5.0) 
            
            if negative_pixels < min_artifact_score:
                min_artifact_score = negative_pixels
                best_offset = (dx, dy)
                print(f"New best: dx={dx}, dy={dy}, score={negative_pixels}")
                
    print(f"Final Best Offset: dx={best_offset[0]}, dy={best_offset[1]}")
    
except Exception as e:
    print(f"Error: {e}")
