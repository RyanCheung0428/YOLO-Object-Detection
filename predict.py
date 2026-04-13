from __future__ import annotations

import argparse
from pathlib import Path

from web.yolo_utils import find_best_weights, run_predictions

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    # CLI supports batch prediction and optional custom output/weights path.
    parser = argparse.ArgumentParser(
        description="Use a trained YOLOv8 model to detect fruits in one or more images."
    )
    parser.add_argument(
        "--image",
        nargs="+",
        required=True,
        help="One or more input images.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to trained weights. Defaults to the latest best.pt under runs/.",
    )
    parser.add_argument(
        "--output-dir",
        default="prediction_outputs",
        help="Directory to save annotated prediction images.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detection.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=896,
        help="Inference image size. Use the same size as training for best recall.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold for filtering overlapping detections.",
    )
    return parser.parse_args()


def resolve_cli_path(path_arg: str) -> Path:
    # Resolve relative input against project root for predictable CLI behavior.
    path = Path(path_arg)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    image_paths = [resolve_cli_path(path) for path in args.image]
    # Fallback to latest available best.pt when --weights is not provided.
    weights_path = resolve_cli_path(args.weights) if args.weights else find_best_weights()
    output_dir = resolve_cli_path(args.output_dir)

    print(f"Using weights: {weights_path.resolve()}")
    print(f"Predicting {len(image_paths)} image(s)...")
    predictions = run_predictions(
        image_paths=image_paths,
        weights_path=weights_path,
        output_dir=output_dir,
        conf=args.conf,
        imgsz=args.imgsz,
        iou=args.iou,
    )

    for item in predictions:
        print(f"\nImage: {item['image_path'].resolve()}")
        detections = item["detections"]
        if not detections:
            print("No objects detected.")
        else:
            for index, detection in enumerate(detections, start=1):
                print(
                    f"{index}. class={detection['class_name']}, "
                    f"bbox={detection['bbox']}, "
                    f"confidence={detection['confidence']}"
                )

        saved_path = item.get("saved_path")
        if saved_path is not None:
            print(f"Saved prediction image to: {saved_path.resolve()}")


if __name__ == "__main__":
    main()
