import os
import sys
import urllib.request
from PIL import Image, ImageOps
import numpy as np

# Configuration
MASK_URLS = {
    48: "https://raw.githubusercontent.com/journey-ad/gemini-watermark-remover/main/src/assets/bg_48.png",
    96: "https://raw.githubusercontent.com/journey-ad/gemini-watermark-remover/main/src/assets/bg_96.png"
}
CACHE_DIR = os.path.expanduser("~/.gemini/assets/masks")

def get_mask_path(size):
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    path = os.path.join(CACHE_DIR, f"bg_{size}.png")
    if not os.path.exists(path):
        print(f"Downloading {size}px mask from GitHub...")
        try:
            urllib.request.urlretrieve(MASK_URLS[size], path)
        except Exception as e:
            print(f"Error downloading mask: {e}")
            sys.exit(1)
    return path

def calculate_alpha(mask_img):
    # Convert to RGB if not already
    if mask_img.mode != 'RGB':
        mask_img = mask_img.convert('RGB')
    
    # Convert to numpy array
    mask_arr = np.array(mask_img).astype(float)
    
    # Alpha is the max of R, G, B channels normalized to 0-1
    # formula: alpha = max(r, g, b) / 255.0
    alpha_map = np.max(mask_arr, axis=2) / 255.0
    
    return alpha_map

def scan_for_watermark(img_pixels, mask_arr, search_margin=200):
    """
    Scans the bottom-right corner of the image for the given mask.
    Returns (min_diff, best_x, best_y)
    """
    ih, iw, _ = img_pixels.shape
    mh, mw, _ = mask_arr.shape
    
    # Define search range (bottom-right corner)
    start_y = max(0, ih - mh - search_margin)
    start_x = max(0, iw - mw - search_margin)
    
    best_x, best_y = iw - mw, ih - mh
    min_diff = float('inf')
    
    # Optimization: Check corners first (most likely locations)
    # Then do a sparse scan, then refine? 
    # For now, full scan is fast enough on small search area (200x200)
    
    # To speed up python loop, we can use slice broadcasting if search area is small
    # But nested loop is clearer. Let's optimize by checking every 2nd pixel first?
    # Or just keep it simple. 200x200 = 40k checks. Fast enough.
    
    for y in range(start_y, ih - mh + 1):
        for x in range(start_x, iw - mw + 1):
            patch = img_pixels[y:y+mh, x:x+mw, :3]
            # Mean Absolute Difference
            diff = np.mean(np.abs(patch - mask_arr))
            
            if diff < min_diff:
                min_diff = diff
                best_x, best_y = x, y
                
    return min_diff, best_x, best_y

def process_image(image_path, output_path):
    print(f"Processing: {image_path}")
    
    try:
        img = Image.open(image_path).convert("RGBA")
    except IOError:
        print("Cannot open image.")
        return

    pixels = np.array(img).astype(float)
    ih, iw, _ = pixels.shape
    
    print(f"Image Size: {iw}x{ih}")
    
    # Strategy: Try BOTH 48px and 96px masks, pick the best match.
    mask_sizes = [48, 96]
    best_candidate = None
    lowest_score = float('inf')
    
    print("Detecting watermark size and position...")
    
    for size in mask_sizes:
        mask_path = get_mask_path(size)
        try:
            mask_img = Image.open(mask_path).convert("RGB")
            mask_arr = np.array(mask_img).astype(float)
            
            # Scan
            score, x, y = scan_for_watermark(pixels, mask_arr)
            print(f"  - Testing {size}px mask: Match Score = {score:.4f} at ({x}, {y})")
            
            if score < lowest_score:
                lowest_score = score
                best_candidate = {
                    'size': size,
                    'mask_arr': mask_arr,
                    'x': x,
                    'y': y
                }
        except Exception as e:
            print(f"  - Failed to test {size}px mask: {e}")

    if not best_candidate:
        print("Could not detect any watermark.")
        return

    print(f"Selected Best Match: {best_candidate['size']}px mask at ({best_candidate['x']}, {best_candidate['y']})")
    
    # Perform Restoration
    mask_arr = best_candidate['mask_arr']
    bx, by = best_candidate['x'], best_candidate['y']
    mh, mw, _ = mask_arr.shape
    
    # Alpha Map
    alpha_map = np.max(mask_arr, axis=2) / 255.0
    alpha_expanded = alpha_map[:, :, np.newaxis]
    
    # Extract Patch
    patch_rgba = pixels[by:by+mh, bx:bx+mw]
    patch_rgb = patch_rgba[:, :, :3]
    
    # Math: O = (I - M) / (1 - A)
    numerator = np.maximum(patch_rgb - mask_arr, 0)
    denominator = np.maximum(1.0 - alpha_expanded, 0.01)
    
    restored_rgb = np.clip(numerator / denominator, 0, 255)
    
    # Write back
    pixels[by:by+mh, bx:bx+mw, :3] = restored_rgb
    
    result_img = Image.fromarray(pixels.astype(np.uint8))
    
    if not output_path:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_clean{ext}"
    
    result_img.save(output_path)
    print(f"Done. Saved to {output_path}")
    
    result_img = Image.fromarray(pixels.astype(np.uint8))
    
    if not output_path:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_clean{ext}"
    
    result_img.save(output_path)
    print(f"Done. Saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python remover.py <image_path> [output_path]")
        sys.exit(1)
    
    img_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    process_image(img_path, out_path)
