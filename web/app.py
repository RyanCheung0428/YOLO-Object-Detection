from __future__ import annotations

import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from .yolo_utils import RESULTS_DIR, detect_and_save, get_model, resolve_weights_path

WEB_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_ROOT.parent
DEFAULT_WEIGHTS_PATH = "runs/fruit_detector/weights/best.pt"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB upload size limit.


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def is_allowed_file(filename: str) -> bool:
    suffix = Path(filename).suffix.lower()
    return suffix in ALLOWED_EXTENSIONS


@app.route("/.well-known/appspecific/com.chrome.devtools.json")
def chrome_devtools_manifest() -> tuple[dict[str, str], int]:
    return {}, 200


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        error_message=None,
        results=[],
        used_weights=DEFAULT_WEIGHTS_PATH,
    )


@app.route("/predict", methods=["POST"])
def predict():
    ensure_dirs()

    uploaded_files = [
        file
        for file in request.files.getlist("image")
        if file is not None and file.filename
    ]
    if not uploaded_files:
        return render_template(
            "index.html",
            error_message="請先選擇至少一張圖片再送出。",
            results=[],
            used_weights=DEFAULT_WEIGHTS_PATH,
        ), 400

    invalid_files = [file.filename for file in uploaded_files if not is_allowed_file(file.filename)]
    if invalid_files:
        return render_template(
            "index.html",
            error_message="只支援 jpg/jpeg/png/webp/bmp 格式。",
            results=[],
            used_weights=DEFAULT_WEIGHTS_PATH,
        ), 400

    requested_weights = request.form.get("weights_path", "").strip() or DEFAULT_WEIGHTS_PATH

    try:
        weights_path = resolve_weights_path(requested_weights)
        model = get_model(weights_path)
        results_payload: list[dict[str, object]] = []

        for uploaded_file in uploaded_files:
            temp_file_path: Path | None = None
            try:
                suffix = Path(secure_filename(uploaded_file.filename)).suffix.lower() or ".jpg"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    uploaded_file.save(temp_file)
                    temp_file_path = Path(temp_file.name).resolve()

                result_image_path, detections = detect_and_save(model, temp_file_path, RESULTS_DIR)
                results_payload.append(
                    {
                        "source_name": uploaded_file.filename,
                        "result_image_url": f"/results/{result_image_path.name}",
                        "detections": detections,
                    }
                )
            finally:
                if temp_file_path and temp_file_path.exists():
                    temp_file_path.unlink(missing_ok=True)

        return render_template(
            "index.html",
            error_message=None,
            results=results_payload,
            used_weights=str(weights_path.relative_to(PROJECT_ROOT)),
        )
    except ValueError:
        return render_template(
            "index.html",
            error_message="權重路徑格式不正確。",
            results=[],
            used_weights=requested_weights,
        ), 400
    except FileNotFoundError as exc:
        return render_template(
            "index.html",
            error_message=str(exc),
            results=[],
            used_weights=requested_weights,
        ), 400
    except Exception:
        return render_template(
            "index.html",
            error_message="圖片檢測失敗，請確認模型與輸入圖片後再試一次。",
            results=[],
            used_weights=requested_weights,
        ), 500


@app.route("/results/<path:filename>", methods=["GET"])
def serve_result_image(filename: str):
    return send_from_directory(RESULTS_DIR, filename)
