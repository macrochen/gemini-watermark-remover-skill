import os
import sys
import urllib.request
import argparse
import subprocess
import platform
import glob
from pathlib import Path
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
    if mask_img.mode != 'RGB':
        mask_img = mask_img.convert('RGB')
    mask_arr = np.array(mask_img).astype(float)
    alpha_map = np.max(mask_arr, axis=2) / 255.0
    return alpha_map

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

def get_latest_downloaded_image():
    """Finds the most recently modified image in the user's Downloads folder."""
    downloads_path = str(Path.home() / "Downloads")
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp']
    list_of_files = []
    for ext in extensions:
        list_of_files.extend(glob.glob(os.path.join(downloads_path, ext)))
    
    if not list_of_files:
        return None
    
    latest_file = max(list_of_files, key=os.path.getmtime)
    return latest_file

def process_image(image_path, output_path=None):
    if not image_path:
        print("No image path provided. Searching for the latest download...")
        image_path = get_latest_downloaded_image()
        if not image_path:
            print("No images found in Downloads.")
            return None
    
    print(f"Processing: {image_path}")
    
    try:
        img = Image.open(image_path).convert("RGBA")
    except IOError:
        print(f"Cannot open image: {image_path}")
        return None

    pixels = np.array(img).astype(float)
    ih, iw, _ = pixels.shape
    
    print(f"Image Size: {iw}x{ih}")
    
    logoSize = 48
    marginRight = 32
    marginBottom = 32

    if imageWidth := iw:
        if imageWidth > 1024 and ih > 1024:
            logoSize = 96
            marginRight = 64
            marginBottom = 64
    
    x = iw - marginRight - logoSize
    y = ih - marginBottom - logoSize
    
    mask_path = get_mask_path(logoSize)
    try:
        mask_img = Image.open(mask_path).convert("RGB")
        mask_arr = np.array(mask_img).astype(float)
    except Exception as e:
        print(f"Error loading mask for size {logoSize}px: {e}")
        return None

    bx, by = x, y
    mh, mw, _ = mask_arr.shape
    
    alpha_map = np.max(mask_arr, axis=2) / 255.0
    alpha_expanded = alpha_map[:, :, np.newaxis]
    
    patch_rgba = pixels[by:by+mh, bx:bx+mw]
    patch_rgb = patch_rgba[:, :, :3]
    
    numerator = patch_rgb - (255.0 * alpha_expanded)
    denominator = np.maximum(1.0 - alpha_expanded, 0.01)
    
    restored_rgb = np.clip(numerator / denominator, 0, 255)
    pixels[by:by+mh, bx:bx+mw, :3] = restored_rgb
    
    result_img = Image.fromarray(pixels.astype(np.uint8))
    
    if not output_path:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_clean{ext}"
    
    if os.path.exists(output_path):
        os.remove(output_path)
        
    result_img.save(output_path)
    print(f"Done. Saved to {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove Gemini watermark")
    parser.add_argument("image_path", nargs="?", help="Path to input image (default: latest in Downloads)")
    parser.add_argument("output_path", nargs="?", help="Path to output image")
    parser.add_argument("--copy", action="store_true", help="Copy result to clipboard (macOS only)")
    
    args = parser.parse_args()
    
    final_path = process_image(args.image_path, args.output_path)
    
    if args.copy and final_path:
        copy_to_clipboard(final_path)
