
import cv2
import numpy as np
import os

def check_image(image_path):
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found.")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    print(f"\nAnalysis for {os.path.basename(image_path)}:")
    print(f"  Shape: {img.shape}")
    print(f"  Mean brightness: {np.mean(img):.2f}")
    
    # Check for black rows (corruption)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row_means = np.mean(gray, axis=1)
    black_rows = np.sum(row_means < 10)
    print(f"  Black rows (mean < 10): {black_rows}")
    
    if black_rows > 0:
        black_indices = np.where(row_means < 10)[0]
        # Group consecutive indices
        diffs = np.diff(black_indices)
        breaks = np.where(diffs > 1)[0]
        ranges = []
        start = black_indices[0]
        for b in breaks:
            end = black_indices[b]
            ranges.append((start, end))
            start = black_indices[b+1]
        ranges.append((start, black_indices[-1]))
        
        print(f"  Corruption detected! Black row ranges: {ranges}")
        print(f"  Percentage corrupt: {black_rows / img.shape[0] * 100:.1f}%")
    else:
        print("  No obvious corruption (black bands) detected.")

if __name__ == "__main__":
    # Check a few random images
    import glob
    head_images = sorted(glob.glob("eval_images/head_*.jpg"))
    wrist_images = sorted(glob.glob("eval_images/wrist_*.jpg"))
    
    print(f"Found {len(head_images)} head images and {len(wrist_images)} wrist images.")
    
    # Check first, middle, last
    indices = [0, len(head_images)//2, -1]
    
    print("Checking Head Camera images:")
    for i in indices:
        if i < len(head_images):
            check_image(head_images[i])
            
    print("\nChecking Wrist Camera images:")
    for i in indices:
        if i < len(wrist_images):
            check_image(wrist_images[i])

