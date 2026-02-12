import os
import sys
import urllib.request
import argparse
import subprocess
import platform
from PIL import Image, ImageOps
import numpy as np

# Configuration
MASK_URLS = {
    48: "https://raw.githubusercontent.com/journey-ad/gemini-watermark-remover/main/src/assets/bg_48.png",
    96: "https://raw.githubusercontent.com/journey-ad/gemini-watermark-remover/main/src/assets/bg_96.png"
}
CACHE_DIR = os.path.join(os.path.dirname(__file__), "masks")

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
    
    for y in range(start_y, ih - mh + 1):
        for x in range(start_x, iw - mw + 1):
            patch = img_pixels[y:y+mh, x:x+mw, :3]
            # Mean Absolute Difference
            diff = np.mean(np.abs(patch - mask_arr))
            
            if diff < min_diff:
                min_diff = diff
                best_x, best_y = x, y
                
    return min_diff, best_x, best_y

def copy_to_clipboard(file_path):
    if platform.system() != 'Darwin':
        print("Warning: Clipboard copy is currently only supported on macOS.")
        return

    abs_path = os.path.abspath(file_path)
    
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.jpg', '.jpeg']:
         apple_script = f'set the clipboard to (read (POSIX file "{abs_path}") as JPEG picture)'
    else:
         apple_script = f'set the clipboard to (read (POSIX file "{abs_path}") as «class PNGf»)'
    
    try:
        subprocess.run(["osascript", "-e", apple_script], check=True)
        print("Image copied to clipboard.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to copy to clipboard: {e}")

def process_image(image_path, output_path=None):
    print(f"Processing: {image_path}")
    
    try:
        img = Image.open(image_path).convert("RGBA")
    except IOError:
        print("Cannot open image.")
        return None

    pixels = np.array(img).astype(float)
    ih, iw, _ = pixels.shape
    
    print(f"Image Size: {iw}x{ih}")
    
    # Detect watermark configuration based on image size (TypeScript rules)
    imageWidth = iw
    imageHeight = ih

    logoSize = 48
    marginRight = 32
    marginBottom = 32

    if imageWidth > 1024 and imageHeight > 1024:
        logoSize = 96
        marginRight = 64
        marginBottom = 64
    
    x = imageWidth - marginRight - logoSize
    y = imageHeight - marginBottom - logoSize
    width = logoSize
    height = logoSize

    print(f"Detected watermark config: size={logoSize}, position=({x}, {y}), margins=({marginRight}, {marginBottom})")

    # Load the specific mask based on determined logoSize
    mask_path = get_mask_path(logoSize)
    try:
        mask_img = Image.open(mask_path).convert("RGB")
        mask_arr = np.array(mask_img).astype(float)
    except Exception as e:
        print(f"Error loading mask for size {logoSize}px: {e}")
        return None

    # Now define best_candidate based on this deterministic detection
    best_candidate = {
        'size': logoSize,
        'mask_arr': mask_arr,
        'x': x,
        'y': y
    }
    
    # Perform Restoration
    mask_arr = best_candidate['mask_arr'] # This mask_arr is used for shape and alpha map calc
    bx, by = best_candidate['x'], best_candidate['y']
    mh, mw, _ = mask_arr.shape
    
    # Alpha Map
    alpha_map = np.max(mask_arr, axis=2) / 255.0
    alpha_expanded = alpha_map[:, :, np.newaxis]
    
    # Define 'M' (logo) as pure white (255, 255, 255)
    logo_white = np.full((mh, mw, 3), 255.0) 

    # Extract Patch
    patch_rgba = pixels[by:by+mh, bx:bx+mw]
    patch_rgb = patch_rgba[:, :, :3]
    
    # Math: O = (I - M) / (1 - A)
    numerator = patch_rgb - (255.0 * alpha_expanded)
    denominator = np.maximum(1.0 - alpha_expanded, 0.01)
    
    restored_rgb = np.clip(numerator / denominator, 0, 255)
    
    # Write back
    pixels[by:by+mh, bx:bx+mw, :3] = restored_rgb
    
    result_img = Image.fromarray(pixels.astype(np.uint8))
    
    if not output_path:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_clean{ext}"
    
    # Explicitly remove existing file to ensure overwrite, as requested by user.
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Removed existing output file: {output_path}")
        
    result_img.save(output_path)
    print(f"Done. Saved to {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove Gemini watermark")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("output_path", nargs="?", help="Path to output image")
    parser.add_argument("--copy", action="store_true", help="Copy result to clipboard (macOS only)")
    
    args = parser.parse_args()
    
    final_path = process_image(args.image_path, args.output_path)
    
    if args.copy and final_path:
        copy_to_clipboard(final_path)
