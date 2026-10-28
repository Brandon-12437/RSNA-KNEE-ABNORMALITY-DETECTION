"""MRI preprocessing and EfficientNetB0 feature extraction."""

import os

import numpy as np
import pydicom
import tensorflow as tf

from PIL import Image


def process_series(series_path, num_slices=8, image_size=(224, 224)):
    """Load, sort, and preprocess DICOM slices from one MRI series."""

    if not os.path.isdir(series_path):
        raise ValueError(f"Series directory not found: {series_path}")

    files = [
        os.path.join(series_path, filename)
        for filename in os.listdir(series_path)
        if filename.lower().endswith(".dcm")
    ]

    if not files:
        raise ValueError(f"No DICOM files found in: {series_path}")

    slices = [pydicom.dcmread(path) for path in files]

    orientation = np.asarray(
        slices[0].ImageOrientationPatient,
        dtype=np.float64,
    )

    row_direction = orientation[:3]
    column_direction = orientation[3:]
    slice_direction = np.cross(row_direction, column_direction)

    positions = []

    for ds in slices:
        position = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
        positions.append(np.dot(position, slice_direction))

    slices = [
        ds for _, ds in sorted(zip(positions, slices), key=lambda item: item[0])
    ]

    if len(slices) > num_slices:
        indices = np.linspace(
            0, len(slices) - 1, num_slices, dtype=int
        )
        slices = [slices[index] for index in indices]

    processed = []

    for ds in slices:
        image = ds.pixel_array.astype(np.float32)

        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        image = image * slope + intercept

        minimum = image.min()
        maximum = image.max()

        if maximum > minimum:
            image = (image - minimum) / (maximum - minimum)
        else:
            image = np.zeros_like(image)

        image = Image.fromarray(
            (image * 255).astype(np.uint8)
        ).resize(image_size, Image.Resampling.BILINEAR)

        processed.append(np.asarray(image, dtype=np.float32) / 255.0)

    return np.asarray(processed, dtype=np.float32)


def extract_study_features(series_directories, base_model, max_slices=8):
    """
    Extract one 1280-dimensional feature vector for an MRI study.

    Parameters
    ----------
    series_directories : list[str]
        Paths to the individual MRI series directories belonging to a study.
    base_model : tensorflow.keras.Model
        Frozen EfficientNetB0 feature extractor.
    max_slices : int
        Maximum number of slices selected from each series.
    """

    if not series_directories:
        raise ValueError("No MRI series directories were provided.")

    selected_slices = []

    for series_path in series_directories:
        images = process_series(
            series_path,
            num_slices=max_slices,
            image_size=(224, 224),
        )

        if len(images) == 0:
            continue

        selected_slices.extend(images)

    if not selected_slices:
        raise ValueError("No MRI slices could be processed.")

    # Convert grayscale images to three channels.
    images = np.asarray(selected_slices, dtype=np.float32)
    images = np.repeat(images[..., np.newaxis], 3, axis=-1)

    # Match the input scaling used by the Kaggle notebook.
    images = images * 255.0

    feature_batches = []

    for start in range(0, len(images), 4):
        batch = tf.convert_to_tensor(images[start:start + 4])
        batch_features = base_model(batch, training=False)
        feature_batches.append(batch_features.numpy())

    features = np.concatenate(feature_batches, axis=0)

    # Average slice features to represent the complete study.
    study_features = features.mean(axis=0)

    if study_features.shape != (1280,):
        raise ValueError(
            f"Expected 1280 features, got {study_features.shape}"
        )

    if not np.isfinite(study_features).all():
        raise ValueError("Extracted features contain invalid values.")

    return study_features

