from pathlib import Path
import cv2
import numpy as np
from PIL import Image

exp_dir = Path("outputs/experiments")
input_crop_path = exp_dir / "anime_crop_240.png"
sr_out_path = exp_dir / "anime_crop_240_sr_x4.png"

# 1. Generate Bicubic x4 baseline
with Image.open(input_crop_path) as img:
    bicubic_x4 = img.resize((img.width * 4, img.height * 4), Image.Resampling.BICUBIC)
    bicubic_path = exp_dir / "anime_crop_240_bicubic_x4.png"
    bicubic_x4.save(bicubic_path)
    print(f"Saved Bicubic x4 baseline: {bicubic_path}")

# 2. Compute Laplacian variance (sharpness metric)
img_bicubic = cv2.imread(str(bicubic_path), cv2.IMREAD_GRAYSCALE)
img_sr = cv2.imread(str(sr_out_path), cv2.IMREAD_GRAYSCALE)

var_bicubic = cv2.Laplacian(img_bicubic, cv2.CV_64F).var()
var_sr = cv2.Laplacian(img_sr, cv2.CV_64F).var()

print(f"Sharpness (Laplacian Variance):")
print(f"  - Bicubic x4:            {var_bicubic:.2f}")
print(f"  - RealBasicVSR (T=1) x4: {var_sr:.2f}")

# 3. Create a side-by-side comparison image
bicubic_rgb = cv2.imread(str(bicubic_path))
sr_rgb = cv2.imread(str(sr_out_path))

# Put label text on images
h, w = bicubic_rgb.shape[:2]
labeled_bicubic = bicubic_rgb.copy()
labeled_sr = sr_rgb.copy()

cv2.putText(labeled_bicubic, "Bicubic x4", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
cv2.putText(labeled_sr, "RealBasicVSR (T=1) x4", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

# Side-by-side composite
composite = np.hstack([labeled_bicubic, labeled_sr])
composite_path = exp_dir / "comparison_bicubic_vs_realbasicvsr_t1.png"
cv2.imwrite(str(composite_path), composite)
print(f"Saved side-by-side comparison: {composite_path}")

# 4. Zoom around the eye in the 960x960 output of the face crop.
zoom_box = (230, 560, 480, 810)
zoom_bicubic = labeled_bicubic[zoom_box[1]:zoom_box[3], zoom_box[0]:zoom_box[2]]
zoom_sr = labeled_sr[zoom_box[1]:zoom_box[3], zoom_box[0]:zoom_box[2]]
zoom_comp = np.hstack([zoom_bicubic, zoom_sr])
zoom_path = exp_dir / "comparison_zoom_detail.png"
cv2.imwrite(str(zoom_path), zoom_comp)
print(f"Saved zoom detail comparison: {zoom_path}")
