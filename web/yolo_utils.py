from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BEST_WEIGHTS = PROJECT_ROOT / "runs" / "fruit_detector" / "weights" / "best.pt"


def load_yolo():
    # Import lazily so CLI help still works even before dependencies are installed.
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: ultralytics\n"
            "Install it with: pip install -r requirements.txt"
        ) from exc
    return YOLO


def get_training_device() -> str:
    # Prefer CUDA when available; otherwise continue on CPU with explicit messaging.
    try:
        import torch
    except ModuleNotFoundError:
        print("PyTorch not found. Training will run only after installing dependencies.")
        return "cpu"

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {device_name} (cuda:0)")
        return "cuda:0"

    print("CUDA is not available. Falling back to CPU.")
    return "cpu"


def prepare_data_yaml(data_yaml: Path) -> Path:
    # Normalize split paths into an absolute data.local.yaml for robust training runs.
    resolved_yaml = data_yaml if data_yaml.is_absolute() else (PROJECT_ROOT / data_yaml)
    resolved_yaml = resolved_yaml.resolve()

    with resolved_yaml.open("r", encoding="utf-8") as file:
        config: dict[str, Any] = yaml.safe_load(file)

    dataset_root = resolved_yaml.parent
    split_dirs = {"train": "train", "val": "valid", "test": "test"}

    def resolve_split(split_name: str) -> str:
        raw_value = config.get(split_name)
        if not raw_value:
            return raw_value
        candidate = (dataset_root / raw_value).resolve()
        if candidate.exists():
            return str(candidate)
        # Fallback keeps compatibility with common train/valid/test folder layouts.
        fallback_dir = split_dirs.get(split_name, split_name)
        return str((dataset_root / fallback_dir / "images").resolve())

    normalized = dict(config)
    normalized["train"] = resolve_split("train")
    normalized["val"] = resolve_split("val")
    normalized["test"] = resolve_split("test")

    generated_yaml = dataset_root / "data.local.yaml"
    with generated_yaml.open("w", encoding="utf-8") as file:
        yaml.safe_dump(normalized, file, allow_unicode=False, sort_keys=False)

    return generated_yaml


def print_dataset_summary(data_yaml: Path) -> None:
    # Print class index mapping to make training logs easier to validate.
    with data_yaml.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    print("Dataset classes:")
    for index, name in enumerate(config.get("names", [])):
        print(f"  {index}: {name}")


def find_best_weights() -> Path:
    # Preserve expected lookup order: canonical run first, then newest run artifact.
    if DEFAULT_BEST_WEIGHTS.exists():
        return DEFAULT_BEST_WEIGHTS

    candidates = sorted(
        (PROJECT_ROOT / "runs").rglob("best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        "No trained model found. Expected best.pt under runs/. "
        "Run train.py first."
    )


def save_prediction_image(result, output_path: Path) -> Path:
    # YOLO plot output is BGR array; convert to RGB before writing with Pillow.
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: pillow\n"
            "Install it with: pip install -r requirements.txt"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotted = result.plot()
    image = Image.fromarray(plotted[..., ::-1])
    image.save(output_path)
    return output_path


def resolve_project_path(path: Path | str) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return (PROJECT_ROOT / resolved).resolve()


def _normalize_model_path(weights_path: Path | None) -> Path:
    resolved_weights = weights_path or find_best_weights()
    return resolve_project_path(resolved_weights)


def _normalize_image_path(image_path: Path | str) -> Path:
    return resolve_project_path(image_path)


def _build_output_path(output_dir: Path | str, image_path: Path, index: int) -> Path:
    # Include index + short UUID to avoid collisions in batch prediction outputs.
    resolved_dir = resolve_project_path(output_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)
    suffix = image_path.suffix or ".jpg"
    unique = uuid.uuid4().hex[:8]
    return resolved_dir / f"{image_path.stem}_{index:02d}_{unique}{suffix}"


def _predict_one(
    model,
    image_path: Path,
    conf: float,
    imgsz: int | None,
    iou: float,
) -> tuple[list[dict[str, Any]], Any]:
    # Keep per-image prediction logic isolated for reuse in batch and single-image flows.
    predict_kwargs: dict[str, Any] = {
        "source": str(image_path),
        "save": False,
        "conf": conf,
        "iou": iou,
    }
    if imgsz is not None:
        predict_kwargs["imgsz"] = imgsz

    results = model.predict(**predict_kwargs)
    result = results[0]

    detections: list[dict[str, Any]] = []
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        detections.append(
            {
                "class_id": class_id,
                "class_name": result.names[class_id],
                "bbox": [round(value) for value in box.xyxy[0].tolist()],
                "confidence": round(box.conf[0].item(), 4),
            }
        )

    return detections, result


def run_predictions(
    image_paths: Sequence[Path | str],
    weights_path: Path | None = None,
    output_dir: Path | str | None = None,
    conf: float = 0.1,
    imgsz: int | None = None,
    iou: float = 0.7,
) -> list[dict[str, Any]]:
    # Main inference entrypoint used by both CLI and web UI.
    YOLO = load_yolo()
    resolved_weights = _normalize_model_path(weights_path)

    if not resolved_weights.exists():
        raise FileNotFoundError(f"Model weights not found: {resolved_weights}")

    model = YOLO(str(resolved_weights))
    results: list[dict[str, Any]] = []

    for index, image_path in enumerate(image_paths, start=1):
        resolved_image = _normalize_image_path(image_path)
        if not resolved_image.exists():
            raise FileNotFoundError(f"Image not found: {resolved_image}")

        detections, result = _predict_one(model, resolved_image, conf, imgsz, iou)
        saved_path = None
        if output_dir is not None:
            saved_path = save_prediction_image(
                result,
                _build_output_path(output_dir, resolved_image, index),
            )

        results.append(
            {
                "image_path": resolved_image,
                "image_name": resolved_image.name,
                "detections": detections,
                "saved_path": saved_path,
            }
        )

    return results


def run_prediction(
    image_path: Path,
    weights_path: Path | None = None,
    output_path: Path | None = None,
    conf: float = 0.1,
    imgsz: int | None = None,
    iou: float = 0.7,
) -> tuple[list[dict[str, Any]], Path | None]:
    # Backward-compatible single-image helper built on top of batch predictions.
    batch_results = run_predictions(
        image_paths=[image_path],
        weights_path=weights_path,
        output_dir=output_path,
        conf=conf,
        imgsz=imgsz,
        iou=iou,
    )[0]
    return batch_results["detections"], batch_results["saved_path"]