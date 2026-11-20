
import os
import numpy as np
import joblib

from pathlib import Path
from PIL import Image

from skimage.feature import hog
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ==========================================
# 1. Project Configuration
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

TRAIN_DIR = BASE_DIR / "data" / "processed" / "train"
MODEL_DIR = BASE_DIR / "models"

IMAGE_SIZE = (128, 128)

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. Feature Extraction
# ==========================================

def extract_features(image_path):

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    # HOG Feature Extraction
    gray_image = np.array(image.convert("L"))

    hog_features = hog(
        gray_image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )

    # HSV Color Histogram
    hsv_image = np.array(image.convert("HSV"))

    hsv_features = []

    for channel in range(3):

        histogram, _ = np.histogram(
            hsv_image[:, :, channel],
            bins=32,
            range=(0, 256)
        )

        histogram = histogram.astype("float32")
        histogram /= (histogram.sum() + 1e-7)

        hsv_features.extend(histogram)

    # Combine HOG and HSV features
    combined_features = np.concatenate([
        hog_features,
        np.array(hsv_features)
    ])

    return combined_features


# ==========================================
# 3. Load Training Dataset
# ==========================================

def load_training_data(dataset_path):

    X = []
    y = []

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Training directory not found: {dataset_path}"
        )

    class_names = sorted([
        folder.name
        for folder in dataset_path.iterdir()
        if folder.is_dir()
    ])

    if not class_names:
        raise ValueError("No fruit class folders found!")

    print("\nLoading training dataset...\n")

    for class_name in class_names:

        class_path = dataset_path / class_name
        image_count = 0

        for image_path in class_path.iterdir():

            if image_path.suffix.lower() not in [
                ".jpg", ".jpeg", ".png", ".bmp", ".webp"
            ]:
                continue

            try:
                features = extract_features(image_path)

                X.append(features)
                y.append(class_name)

                image_count += 1

            except Exception as error:
                print(
                    f"Skipped {image_path.name}: {error}"
                )

        print(f"{class_name}: {image_count} images")

        if image_count == 0:
            raise ValueError(
                f"No valid images found in {class_name}"
            )

    if len(set(y)) < 2:
        raise ValueError(
            "SVM requires at least two different classes."
        )

    return np.array(X), np.array(y), class_names


# ==========================================
# 4. Prepare Training Features
# ==========================================

X_train, y_train, class_names = load_training_data(
    TRAIN_DIR
)

print("\nTraining data loaded successfully!")

print("Number of images:", len(X_train))
print("Number of features:", X_train.shape[1])
print("Classes:", class_names)


# ==========================================
# 5. Create SVM Model
# ==========================================

svm_model = Pipeline([
    ("scaler", StandardScaler()),

    ("classifier", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale"
    ))
])


# ==========================================
# 6. Train Baseline Model
# ==========================================

print("\nTraining SVM Baseline Model...\n")

svm_model.fit(X_train, y_train)

print("Training completed successfully!")


# ==========================================
# 7. Save Trained Model
# ==========================================

model_path = MODEL_DIR / "baseline_svm_model.joblib"

joblib.dump(
    {
        "model": svm_model,
        "classes": class_names,
        "image_size": IMAGE_SIZE
    },
    model_path
)

print("\nBaseline model saved successfully!")
print("Model path:", model_path)

print("\nBaseline implementation completed!")
