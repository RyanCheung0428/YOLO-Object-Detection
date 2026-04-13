from __future__ import annotations

import argparse
from pathlib import Path

from web.yolo_utils import (
    get_training_device,
    load_yolo,
    prepare_data_yaml,
    print_dataset_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parent
# Default training configuration used by CLI unless overridden.
DEFAULT_DATASET_NAME = "dataset"
DEFAULT_DATA_YAML_NAME = "data.yaml"
DEFAULT_MODEL = "yolo11m.pt"
EPOCHS = 300
IMAGE_SIZE = 896
BATCH_SIZE = 12
WORKERS = 12
PATIENCE = 50
CACHE_MODE = "disk"  # Options: 'ram', 'disk', False
RUN_NAME = "runs/fruit_detector_v2"


def parse_args() -> argparse.Namespace:
    # Keep entrypoint focused on CLI parsing and orchestration choices.
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 with a selectable dataset folder or YAML config."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_NAME,
        help=(
            "Dataset folder under project root. "
            "Example: dataset or dataset2. Ignored when --data-yaml is used."
        ),
    )
    parser.add_argument(
        "--data-yaml",
        default=None,
        help=(
            "Path to a data.yaml file (absolute or project-relative). "
            "Overrides --dataset when provided."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "YOLO model weights name or path. "
            "Examples: yolo11s.pt, yolo11m.pt, yolo11x.pt, runs/fruit_detector/weights/best.pt"
        ),
    )
    return parser.parse_args()

def resolve_data_yaml(dataset_name: str, data_yaml_override: str | None) -> Path:
    # Prefer explicit YAML when provided; otherwise derive from dataset folder.
    if data_yaml_override:
        override_path = Path(data_yaml_override)
        if not override_path.is_absolute():
            override_path = PROJECT_ROOT / override_path
        return override_path.resolve()

    return (PROJECT_ROOT / dataset_name / DEFAULT_DATA_YAML_NAME).resolve()


def resolve_model_source(model_name_or_path: str) -> str:
    # Resolve local paths first, then pass through canonical model names for auto-download.
    candidate = Path(model_name_or_path)
    if candidate.is_absolute():
        return str(candidate.resolve())

    project_candidate = (PROJECT_ROOT / candidate).resolve()
    if project_candidate.exists():
        return str(project_candidate)

    # Keep canonical Ultralytics model names (for auto-download), e.g. yolov8l.pt.
    return model_name_or_path


def train_and_validate(data_yaml: Path, model_source: str) -> tuple[Path, Path]:
    # Shared utility handles optional dependency import with actionable errors.
    YOLO = load_yolo()
    normalized_yaml = prepare_data_yaml(data_yaml)
    device = get_training_device()
    print(f"Using dataset config: {normalized_yaml}")
    print_dataset_summary(normalized_yaml)

    model = YOLO(model_source)
    # Keep training defaults centralized in constants for easy tuning.
    train_results = model.train(
        data=str(normalized_yaml),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=str(PROJECT_ROOT),
        name=RUN_NAME,
        exist_ok=True,
        device=device,
        workers=WORKERS,
        cache=CACHE_MODE,
        patience=PATIENCE,
    )
    print("Training completed.")

    val_results = model.val(data=str(normalized_yaml), device=device)
    print("Validation completed.")
    print(val_results)

    save_dir = Path(train_results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Best weights not found: {best_weights}")

    return best_weights, save_dir


def main() -> None:
    args = parse_args()
    data_yaml = resolve_data_yaml(args.dataset, args.data_yaml)
    model_source = resolve_model_source(args.model)
    # Provide a clear next step when dataset path resolution fails.
    if not data_yaml.exists():
        if args.data_yaml:
            raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_yaml}\n"
            "Try: python train.py --dataset dataset2\n"
            "Or:  python train.py --data-yaml dataset2/data.yaml"
        )

    print(f"Selected dataset config: {data_yaml}")
    print(f"Selected model: {model_source}")
    best_weights, save_dir = train_and_validate(data_yaml, model_source)
    print(f"Training outputs saved to: {save_dir.resolve()}")
    print(f"Best model weights: {best_weights.resolve()}")


if __name__ == "__main__":
    main()
