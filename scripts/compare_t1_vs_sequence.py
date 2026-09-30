import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

exp_dir = Path("outputs/experiments")
seq_frame0_in = exp_dir / "test_sequence_frames/frame_000.png"
seq_frame0_out = exp_dir / "sequence_sr_x4/frame_000.png"

# Run single T=1 on seq_frame0_in
from backend.app.inference.realbasicvsr_pipeline import infer_rb_single, load_realbasicvsr, read_image_rgb, save_image_rgb
import torch

net, device = load_realbasicvsr("best.pth", device=torch.device("cpu"))
img_rgb = read_image_rgb(seq_frame0_in)
sr_t1_rgb = infer_rb_single(net, img_rgb, device=device)
sr_t1_path = exp_dir / "frame_000_sr_t1.png"
save_image_rgb(sr_t1_path, sr_t1_rgb)

# Read both outputs
img_seq = cv2.imread(str(seq_frame0_out))
img_t1 = cv2.imread(str(sr_t1_path))

# Compare difference
diff = cv2.absdiff(img_seq, img_t1)
mean_diff = np.mean(diff)
max_diff = np.max(diff)

print(f"Comparison between Sequence SR (T=5) and Single SR (T=1):")
print(f"  - Mean Pixel Difference (0..255): {mean_diff:.2f}")
print(f"  - Max Pixel Difference (0..255):  {max_diff}")

labeled_t1 = img_t1.copy()
labeled_seq = img_seq.copy()
cv2.putText(labeled_t1, "Single T=1 (512x512)", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
cv2.putText(labeled_seq, "Sequence T=5 (512x512)", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

comp_seq_vs_t1 = np.hstack([labeled_t1, labeled_seq])
comp_seq_path = exp_dir / "comparison_t1_vs_sequence.png"
cv2.imwrite(str(comp_seq_path), comp_seq_vs_t1)
print(f"Saved comparison to: {comp_seq_path}")
