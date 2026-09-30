from pathlib import Path
from PIL import Image

anime_path = Path("assets/anime_before.jpg")
exp_dir = Path("outputs/experiments")
exp_dir.mkdir(parents=True, exist_ok=True)

# Asuka face crop box: x 370..594 (224), y 560..784 (224)
crop_box = (370, 560, 594, 784) # 224x224 (divisible by 32: 224 / 32 = 7)
with Image.open(anime_path) as img:
    face_crop = img.crop(crop_box)
    face_crop.save(exp_dir / "asuka_face_224.png")
    print(f"Saved Asuka face crop: {face_crop.size} to {exp_dir / 'asuka_face_224.png'}")
