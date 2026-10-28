import os
import json

import joblib
import numpy as np

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.applications import EfficientNetB0

from model.feature_extraction import extract_study_features


# --------------------------------------------------
# 1. Application configuration
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

with open(os.path.join(MODEL_DIR, "model_config.json")) as file:
    CONFIG = json.load(file)

classifier = joblib.load(
    os.path.join(MODEL_DIR, "logistic_regression.joblib")
)

TARGET_COLUMNS = CONFIG["target_columns"]
IMAGE_SIZE = tuple(CONFIG["image_size"])

app = Flask(__name__)

# Limit uploads to 500 MB for this local prototype.
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024


# --------------------------------------------------
# 2. Load EfficientNetB0
# --------------------------------------------------

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    pooling="avg",
)

base_model.trainable = False


# --------------------------------------------------
# 3. Basic health check
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "RSNA Knee MRI Prediction API",
        "status": "running",
        "number_of_labels": len(TARGET_COLUMNS),
    })


# --------------------------------------------------
# 4. Prediction endpoint
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    """
    Upload one ZIP archive containing an MRI study.

    Expected structure inside the ZIP:

    study.zip
      study_id/
        series_id_1/
          image1.dcm
          image2.dcm
        series_id_2/
          image1.dcm
    """

    if "file" not in request.files:
        return jsonify({
            "error": "Upload a ZIP file using the 'file' field."
        }), 400

    uploaded_file = request.files["file"]

    if not uploaded_file.filename:
        return jsonify({"error": "No file selected."}), 400

    if not uploaded_file.filename.lower().endswith(".zip"):
        return jsonify({
            "error": "The uploaded file must be a ZIP archive."
        }), 400

    import tempfile
    import zipfile

    with tempfile.TemporaryDirectory() as temporary_directory:
        zip_path = os.path.join(temporary_directory, "study.zip")
        extract_directory = os.path.join(
            temporary_directory, "extracted"
        )
        os.makedirs(extract_directory, exist_ok=True)

        uploaded_file.save(zip_path)

        try:
            with zipfile.ZipFile(zip_path, "r") as archive:
                # Reject unsafe archive paths.
                for member in archive.infolist():
                    destination = os.path.realpath(
                        os.path.join(extract_directory, member.filename)
                    )

                    if os.path.commonpath(
                        [extract_directory, destination]
                    ) != extract_directory:
                        return jsonify({
                            "error": "Unsafe ZIP archive paths detected."
                        }), 400

                archive.extractall(extract_directory)

        except zipfile.BadZipFile:
            return jsonify({
                "error": "The uploaded file is not a valid ZIP archive."
            }), 400

        # Find directories containing DICOM files.
        series_directories = []

        for root, _, filenames in os.walk(extract_directory):
            if any(name.lower().endswith(".dcm") for name in filenames):
                series_directories.append(root)

        if not series_directories:
            return jsonify({
                "error": "No DICOM files were found in the ZIP archive."
            }), 400

        try:
            features = extract_study_features(
                series_directories=series_directories,
                base_model=base_model,
                max_slices=CONFIG["num_slices_per_series"],
            )

            probabilities = classifier.predict_proba(
                features.reshape(1, -1)
            )[0]

        except Exception:
            app.logger.exception("MRI prediction failed")
            return jsonify({
                "error": "The MRI study could not be processed."
            }), 500

        results = {
            label: round(float(probability), 4)
            for label, probability in zip(
                TARGET_COLUMNS, probabilities
            )
        }

        return jsonify({
            "predictions": results,
            "threshold": CONFIG["probability_threshold"],
            "notice": (
                "Experimental research predictions only; "
                "not a medical diagnosis."
            ),
        })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)