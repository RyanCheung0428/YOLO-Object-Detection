from __future__ import annotations

import uuid
from pathlib import Path

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

from web.yolo_utils import find_best_weights, run_predictions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

UPLOAD_DIR = PROJECT_ROOT / "web" / "uploads"
RESULT_DIR = PROJECT_ROOT / "web" / "results"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def allowed_file(filename: str) -> bool:
    # Limit uploads to image extensions expected by the inference pipeline.
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def ensure_dirs() -> None:
    # Runtime directories are created lazily to keep setup friction low.
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    results = []
    weights_path = None

    if request.method == "POST":
        ensure_dirs()
        # Keep only files that are actually present and have a filename.
        files = [file for file in request.files.getlist("images") if file and file.filename]

        if not files:
            error = "Please choose at least one image."
        else:
            invalid_files = [file.filename for file in files if not allowed_file(file.filename)]
            if invalid_files:
                error = "Only jpg, jpeg, png, bmp, and webp files are supported."
            else:
                upload_paths = []
                uploaded_meta = []
                for file in files:
                    # Use secure + unique names to avoid collisions and path injection issues.
                    safe_name = secure_filename(file.filename)
                    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
                    upload_path = UPLOAD_DIR / unique_name
                    file.save(upload_path)
                    upload_paths.append(upload_path)
                    uploaded_meta.append(
                        {
                            "original_name": file.filename,
                            "safe_name": safe_name,
                        }
                    )

                try:
                    weights_path = find_best_weights()
                    # Run batch inference and map each prediction back to uploaded metadata.
                    batch_predictions = run_predictions(
                        image_paths=upload_paths,
                        weights_path=weights_path,
                        output_dir=RESULT_DIR,
                    )
                    for meta, prediction in zip(uploaded_meta, batch_predictions):
                        saved_path = prediction.get("saved_path")
                        results.append(
                            {
                                "original_name": meta["original_name"],
                                "safe_name": meta["safe_name"],
                                "detections": prediction["detections"],
                                "image_url": (
                                    url_for("result_file", filename=saved_path.name)
                                    if saved_path is not None
                                    else None
                                ),
                            }
                        )
                except Exception as exc:
                    # Keep UI error simple/actionable instead of exposing traceback details.
                    error = str(exc)

    return render_template(
        "index.html",
        error=error,
        results=results,
        weights_path=str(weights_path) if weights_path else None,
    )


@app.route("/results/<path:filename>")
def result_file(filename: str):
    from flask import send_from_directory

    # Serve generated result images from the dedicated results directory.
    return send_from_directory(RESULT_DIR.resolve(), filename)


if __name__ == "__main__":
    ensure_dirs()
    app.run(debug=True)
