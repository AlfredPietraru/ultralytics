import os
from pathlib import Path
from tqdm import tqdm
from zipfile import ZipFile
from urllib.request import urlretrieve
from PIL import Image

# URLs for VisDrone dataset
urls = {
    'train': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip',
    'val': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip',
    'test': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip'
}

dataset_dir = Path('VisDrone')
dataset_dir.mkdir(parents=True, exist_ok=True)

# Download and extract
for name, url in urls.items():
    zip_path = dataset_dir / f'{name}.zip'
    print(f'Downloading {name} set...')
    urlretrieve(url, zip_path)
    print(f'Extracting {zip_path}...')
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    zip_path.unlink()  # remove zip file after extraction

def convert_box(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

def visdrone2yolo(dir):
    img_dir = dir / 'images'
    ann_dir = dir / 'annotations'
    label_dir = dir / 'labels'
    label_dir.mkdir(parents=True, exist_ok=True)

    for ann_file in tqdm(list(ann_dir.glob('*.txt')), desc=f'Converting {dir.name}'):
        img_file = img_dir / ann_file.name.replace('.txt', '.jpg')
        if not img_file.exists():
            continue
        img_size = Image.open(img_file).size
        yolo_lines = []
        with open(ann_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split(',')
                if parts[4] == '0':  # ignored region
                    continue
                cls = int(parts[5]) - 1  # classes are 1-indexed in VisDrone
                box = convert_box(img_size, tuple(map(int, parts[:4])))
                yolo_lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
        label_path = label_dir / ann_file.name
        with open(label_path, 'w') as f:
            f.writelines(yolo_lines)

# Convert all splits
for split in ['VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev']:
    visdrone2yolo(dataset_dir / split)

print("✅ Download and conversion complete.")
