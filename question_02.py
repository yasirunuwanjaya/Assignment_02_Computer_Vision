import cv2
import numpy as np

# 1. Load the image to get pixel dimensions
img = cv2.imread("earrings.jpg")

if img is None:
    print("Error: Could not find 'earings.jpg'. Make sure it is in the same folder.")
else:
    # height and width of the image in pixels
    h_px, w_px = img.shape[:2]
    print(f"Image Dimensions: {w_px}px wide by {h_px}px tall")

    # 2. Camera Parameters
    f = 8.0              # Focal length (mm)
    u = 720.0            # Distance to object plane (mm)
    pixel_pitch = 0.0022 # 2.2 micrometers converted to mm

    # 3. Calculations
    # Image distance (v) using thin lens equation
    v = (f * u) / (u - f)
    # Magnification (M)
    M = v / u

    # 4. Convert Pixel dimensions to Real-world dimensions (mm)
    real_width = (w_px * pixel_pitch) / M
    real_height = (h_px * pixel_pitch) / M

    print(f"\n--- Question 2 Results ---")
    print(f"Magnification (M): {M:.6f}")
    print(f"Real Size of Image Area: {real_width:.2f} mm x {real_height:.2f} mm")