import pandas as pd
import os
from PIL import Image

def csv_to_yolo(csv_file, folder):
    print(f"\nProcessing CSV: {csv_file}")
    os.makedirs(os.path.join(folder, "labels"), exist_ok=True)

    df = pd.read_csv(csv_file)
    print(f"Total rows in CSV: {len(df)}")
    
    total_images = len(df['filename'].unique())
    print(f"Total unique images: {total_images}")

    for i, img_file in enumerate(df['filename'].unique(), 1):
        img_path = os.path.join(folder, img_file)
        label_path = os.path.join(folder, "labels", os.path.splitext(img_file)[0] + ".txt")

        if not os.path.exists(img_path):
            print(f"[WARNING] Image not found: {img_path}")
            continue

        with Image.open(img_path) as im:
            w, h = im.size

        boxes = df[df['filename'] == img_file]

        with open(label_path, 'w') as f:
            for _, row in boxes.iterrows():
                x_min, y_min, x_max, y_max = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                class_id = 0  # microplastic

                # Convert to YOLO normalized format
                x_center = (x_min + x_max) / 2 / w
                y_center = (y_min + y_max) / 2 / h
                width = (x_max - x_min) / w
                height = (y_max - y_min) / h

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        if i % 100 == 0:
            print(f"Processed {i}/{total_images} images")

    print(f"Done processing {csv_file}")

# Run for train and valid
csv_to_yolo("dataset/train/_annotations.csv", "dataset/train")
csv_to_yolo("dataset/valid/_annotations.csv", "dataset/valid")