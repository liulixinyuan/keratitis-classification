"""Generate the eight offline training augmentations used by the pipeline."""

import argparse
import os
import random
import shutil

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_transformations():
    return [
        transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(p=1.0),
        ]),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.0)),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.Compose([
            transforms.RandomGrayscale(p=0.5),
            transforms.RandomVerticalFlip(p=1.0),
        ]),
        transforms.RandomPerspective(distortion_scale=0.4, p=1.0),
        transforms.Compose([
            transforms.GaussianBlur(kernel_size=3),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
        ]),
        transforms.Compose([
            transforms.RandomPosterize(bits=3, p=1.0),
            transforms.RandomSolarize(threshold=128, p=1.0),
        ]),
    ]


def augment_dataset(input_dir, output_dir, seed=42, overwrite=False):
    set_seed(seed)
    transformations = build_transformations()
    os.makedirs(output_dir, exist_ok=True)

    class_names = sorted(
        name for name in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, name))
    )
    for class_name in tqdm(class_names, desc="Classes"):
        class_dir = os.path.join(input_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        image_names = sorted(
            name for name in os.listdir(class_dir)
            if name.lower().endswith(IMAGE_EXTENSIONS)
        )
        for image_name in tqdm(image_names, desc=class_name, leave=False):
            image_path = os.path.join(class_dir, image_name)
            output_original = os.path.join(output_class_dir, image_name)
            if overwrite or not os.path.exists(output_original):
                shutil.copy2(image_path, output_original)

            with Image.open(image_path) as image:
                image = image.convert("RGB")
                stem, _ = os.path.splitext(image_name)
                for index, transformation in enumerate(transformations, start=1):
                    output_augmented = os.path.join(output_class_dir, f"{stem}_aug_{index}.jpg")
                    if overwrite or not os.path.exists(output_augmented):
                        transformation(image).save(output_augmented)


def main():
    parser = argparse.ArgumentParser(description="Create reproducible offline training augmentations.")
    parser.add_argument("--input-dir", default=os.path.join(PROJECT_ROOT, "data", "train"))
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "data", "train_transformed"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Training directory not found: {input_dir}")
    augment_dataset(input_dir, output_dir, seed=args.seed, overwrite=args.overwrite)
    print(f"Augmented training dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
