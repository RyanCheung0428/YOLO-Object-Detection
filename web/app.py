from __future__ import annotations

import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from .yolo_utils import RESULTS_DIR, WEIGHTS_PATH, detect_and_save, get_model

WEB_ROOT = Path(__file__).resolve().parent
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
            error_message="Please choose at least one image before submitting.",
            results=[],
        ), 400

    invalid_files = [file.filename for file in uploaded_files if not is_allowed_file(file.filename)]
    if invalid_files:
        return render_template(
            "index.html",
            error_message="Only jpg/jpeg/png/webp/bmp files are supported.",
            results=[],
        ), 400

    try:
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError("Could not find model weights at runs/fruit_detector/weights/best.pt.")

        model = get_model(WEIGHTS_PATH)
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
            "results.html",
            error_message=None,
            results=results_payload,
        )
    except FileNotFoundError as exc:
        return render_template(
            "index.html",
            error_message=str(exc),
            results=[],
        ), 400
    except Exception:
        return render_template(
            "index.html",
            error_message="Image detection failed. Please check the model and input images, then try again.",
            results=[],
        ), 500


@app.route("/results/<path:filename>", methods=["GET"])
def serve_result_image(filename: str):
    return send_from_directory(RESULTS_DIR, filename)
