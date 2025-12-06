
import os
import cv2
import numpy as np

def analyze_image(path, name):
    if not os.path.exists(path):
        print(f"❌ {name} not found at {path}")
        return
    
    img = cv2.imread(path)
    if img is None:
        print(f"❌ Failed to load {name}")
        return

    h, w, c = img.shape
    mean_brightness = np.mean(img)
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    
    print(f"✅ {name}:")
    print(f"   Dimensions: {w}x{h}")
    print(f"   Mean Brightness: {mean_brightness:.1f}")
    print(f"   Blur Score (Laplacian Var): {laplacian_var:.1f} (Lower = More Blurred, <100 is usually blurry)")
    
    # Simple check for black images
    if mean_brightness < 5:
        print(f"  ⚠️  WARNING: {name} is very dark/black!")

analyze_image("/home/jrobot/project/Isaac-GR00T/eval_images/img_01192.jpg", "User Provided Image")
