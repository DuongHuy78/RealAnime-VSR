from pathlib import Path
import cv2
import numpy as np
from PIL import Image

exp_dir = Path("outputs/experiments")
face_in = exp_dir / "asuka_face_224.png"
face_sr = exp_dir / "asuka_face_224_sr_x4.png"

# 1. Bicubic baseline
with Image.open(face_in) as img:
    bicubic_img = img.resize((img.width * 4, img.height * 4), Image.Resampling.BICUBIC)
    bicubic_path = exp_dir / "asuka_face_bicubic_x4.png"
    bicubic_img.save(bicubic_path)

# 2. Sharpness comparison
img_b = cv2.imread(str(bicubic_path), cv2.IMREAD_GRAYSCALE)
img_s = cv2.imread(str(face_sr), cv2.IMREAD_GRAYSCALE)
var_b = cv2.Laplacian(img_b, cv2.CV_64F).var()
var_s = cv2.Laplacian(img_s, cv2.CV_64F).var()

print(f"Face Sharpness (Laplacian Var):")
print(f"  - Bicubic x4:            {var_b:.2f}")
print(f"  - RealBasicVSR (T=1) x4: {var_s:.2f}")

# 3. Create side-by-side comparison
b_rgb = cv2.imread(str(bicubic_path))
s_rgb = cv2.imread(str(face_sr))

labeled_b = b_rgb.copy()
labeled_s = s_rgb.copy()

# Add labels
cv2.putText(labeled_b, "Bicubic x4 (896x896)", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
cv2.putText(labeled_s, "RealBasicVSR T=1 x4 (896x896)", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

comp = np.hstack([labeled_b, labeled_s])
comp_path = exp_dir / "asuka_face_comparison.png"
cv2.imwrite(str(comp_path), comp)
print(f"Saved side-by-side comparison to {comp_path}")
