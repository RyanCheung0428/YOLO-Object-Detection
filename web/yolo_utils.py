from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = WEB_ROOT / "results"

_model_cache: Any | None = None
_model_path_cache: Path | None = None


def load_yolo() -> Any:
    # Import lazily so the app can still start and show actionable dependency errors.
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: ultralytics\n"
            "Install it with: pip install -r requirements.txt"
        ) from exc
    return YOLO


def resolve_weights_path(explicit_path: str | None = None) -> Path:
    if explicit_path:
        explicit_candidate = Path(explicit_path)
        if not explicit_candidate.is_absolute():
            explicit_candidate = (PROJECT_ROOT / explicit_candidate).resolve()
        if explicit_candidate.exists():
            return explicit_candidate

    preferred = (PROJECT_ROOT / "runs" / "fruit_detector" / "weights" / "best.pt").resolve()
    if preferred.exists():
        return preferred

    runs_root = (PROJECT_ROOT / "runs").resolve()
    if runs_root.exists():
        best_files = sorted(
            runs_root.glob("**/best.pt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if best_files:
            return best_files[0].resolve()

    raise FileNotFoundError(
        "Could not find model weights. Expected one of:\n"
        "1) runs/fruit_detector/weights/best.pt\n"
        "2) newest best.pt under runs/\n"
        "You can also provide a custom path in the web form."
    )


def get_model(weights_path: Path) -> Any:
    global _model_cache, _model_path_cache

    resolved_path = weights_path.resolve()
    if _model_cache is None or _model_path_cache != resolved_path:
        YOLO = load_yolo()
        _model_cache = YOLO(str(resolved_path))
        _model_path_cache = resolved_path

    return _model_cache


def summarize_detections(result: Any) -> list[dict[str, Any]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    names = getattr(result, "names", {})
    detections: list[dict[str, Any]] = []

    for box in boxes:
        cls_index = int(box.cls[0].item()) if box.cls is not None else -1
        confidence = float(box.conf[0].item()) if box.conf is not None else 0.0
        xyxy = box.xyxy[0].tolist() if box.xyxy is not None else [0.0, 0.0, 0.0, 0.0]

        if isinstance(names, dict):
            class_name = str(names.get(cls_index, cls_index))
        elif isinstance(names, list) and 0 <= cls_index < len(names):
            class_name = str(names[cls_index])
        else:
            class_name = str(cls_index)

        detections.append(
            {
                "class": class_name,
                "confidence": round(confidence, 4),
                "bbox_xyxy": [round(float(value), 2) for value in xyxy],
            }
        )

    return detections


def detect_and_save(
    model: Any,
    image_path: Path,
    output_dir: Path | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    target_dir = output_dir or RESULTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    results = model.predict(source=str(image_path), save=False, verbose=False)
    if not results:
        raise RuntimeError("No inference result was produced by the model.")

    first_result = results[0]
    output_name = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    output_path = (target_dir / output_name).resolve()

    first_result.save(filename=str(output_path))
    return output_path, summarize_detections(first_result)
