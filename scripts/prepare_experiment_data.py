from pathlib import Path
from PIL import Image

assets_dir = Path("assets")
exp_dir = Path("outputs/experiments")
exp_dir.mkdir(parents=True, exist_ok=True)

# 1. Prepare cropped anime single image
anime_img_path = assets_dir / "anime_before.jpg"
if anime_img_path.is_file():
    with Image.open(anime_img_path) as img:
        # Crop 240x240 around the face and eye, rather than the flat red background.
        crop_box = (360, 550, 600, 790)
        cropped = img.crop(crop_box)
        crop_path = exp_dir / "anime_crop_240.png"
        cropped.save(crop_path)
        print(f"Saved cropped test image: {crop_path} (size: {cropped.size})")

# 2. Prepare a five-frame sequence around the eye with subtle crop shifts.
seq_dir = exp_dir / "test_sequence_frames"
seq_dir.mkdir(parents=True, exist_ok=True)
if anime_img_path.is_file():
    with Image.open(anime_img_path) as img:
        face = img.crop((370, 560, 594, 784))
        for i in range(5):
            shift_box = (20 + i * 2, 80 + i, 148 + i * 2, 208 + i)
            frame = face.crop(shift_box)
            frame.save(seq_dir / f"frame_{i:03d}.png")
        print(f"Saved 5 sequence frames to {seq_dir}")
