import cv2
import numpy as np
import os

image_path = "eval_images/img_01198.jpg"

if not os.path.exists(image_path):
    print(f"Image {image_path} not found.")
else:
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load image.")
    else:
        print(f"Image shape: {img.shape}")
        print(f"Mean brightness: {np.mean(img)}")
        
        # Check for black rows (corruption)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        row_means = np.mean(gray, axis=1)
        black_rows = np.sum(row_means < 10)
        print(f"Black rows (mean < 10): {black_rows}")
        
        if black_rows > 0:
            print("Potential corruption detected: Horizontal black banding.")
            
            # Print distribution of black rows
            black_indices = np.where(row_means < 10)[0]
            print(f"Black row indices (first 10): {black_indices[:10]}")
            print(f"Black row indices (last 10): {black_indices[-10:]}")
