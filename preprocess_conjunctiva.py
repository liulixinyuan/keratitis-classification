"""Generate conjunctiva-only images using a trained YOLO11 segmentation model.

This script reproduces the preprocessing used before downstream classification:
1. run YOLO11 instance-segmentation inference;
2. select the first predicted mask (predictions are confidence-ordered);
3. resize and binarize the mask at 0.5;
4. smooth the boundary, fill holes, remove small regions, and retain the
   largest connected component;
5. set pixels outside the conjunctiva mask to black.

Example:
    python preprocess_conjunctiva.py \
        --input-dir path/to/raw_images \
        --output-dir path/to/conjunctiva_images \
        --weights path/to/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create conjunctiva-only images for classification.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing raw input images.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for preprocessed output images.")
    parser.add_argument("--weights", type=Path, required=True, help="Trained YOLO11 segmentation weights.")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Mask binarization threshold.")
    parser.add_argument("--kernel-size", type=int, default=7, help="Odd Gaussian smoothing kernel size.")
    parser.add_argument("--sigma", type=float, default=2.0, help="Gaussian smoothing sigma.")
    parser.add_argument("--morph-iterations", type=int, default=3, help="Morphological closing iterations.")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG output quality.")
    parser.add_argument("--device", default=None, help="Inference device, e.g. 0 or cpu.")
    return parser.parse_args()


def resize_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize a mask to (height, width) using nearest-neighbor interpolation."""
    if mask.shape[:2] == target_shape:
        return mask
    return cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)


def smooth_mask(
    mask: np.ndarray,
    kernel_size: int = 7,
    sigma: float = 2.0,
    morph_iterations: int = 3,
) -> np.ndarray:
    """Smooth boundaries, fill holes, remove noise, and retain the largest region."""
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("--kernel-size must be a positive odd integer")

    binary = (mask > 0).astype(np.uint8)
    smoothed = cv2.GaussianBlur(binary.astype(np.float32), (kernel_size, kernel_size), sigma)
    _, smoothed = cv2.threshold(smoothed, 0.2, 1.0, cv2.THRESH_BINARY)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    smoothed = cv2.morphologyEx(
        smoothed, cv2.MORPH_CLOSE, close_kernel, iterations=morph_iterations
    )
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, open_kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (smoothed * 255).astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        return smoothed.astype(np.uint8)

    largest_label = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    return (labels == largest_label).astype(np.uint8)


def apply_conjunctiva_mask(
    image: np.ndarray,
    predicted_mask: np.ndarray,
    mask_threshold: float,
    kernel_size: int,
    sigma: float,
    morph_iterations: int,
) -> np.ndarray:
    """Apply the post-processed conjunctiva mask and blacken all other pixels."""
    mask = resize_mask(predicted_mask, image.shape[:2])
    mask = (mask > mask_threshold).astype(np.uint8)
    mask = smooth_mask(mask, kernel_size, sigma, morph_iterations)

    output = np.zeros_like(image)
    output[mask > 0] = image[mask > 0]
    return output


def find_images(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    if not args.weights.is_file():
        raise FileNotFoundError(f"Model weights do not exist: {args.weights}")

    image_paths = find_images(args.input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {args.input_dir}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Ultralytics is required. Install it with: pip install ultralytics") from exc

    model = YOLO(str(args.weights))
    processed = 0
    missed = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipped unreadable image: {image_path}")
            missed += 1
            continue

        prediction = model(str(image_path), device=args.device, verbose=False)[0]
        if prediction.masks is None or len(prediction.masks.data) == 0:
            print(f"No conjunctiva mask detected: {image_path}")
            missed += 1
            continue

        # Preserve the original preprocessing behavior: use the first predicted mask.
        predicted_mask = prediction.masks.data[0].cpu().numpy()
        output = apply_conjunctiva_mask(
            image,
            predicted_mask,
            args.mask_threshold,
            args.kernel_size,
            args.sigma,
            args.morph_iterations,
        )

        relative_path = image_path.relative_to(args.input_dir).with_suffix(".jpg")
        output_path = args.output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), output, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]):
            raise OSError(f"Failed to write output image: {output_path}")
        processed += 1

    print(f"Completed: {processed} images processed; {missed} images skipped or without masks.")


if __name__ == "__main__":
    main()
