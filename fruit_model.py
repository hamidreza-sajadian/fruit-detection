"""Notebook-oriented fruit image classifier using HSV + LBP + shape descriptors and SVM."""
from pathlib import Path
import time
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_SIZE = (160, 160)


def extract_features(image_bgr):
    """Create a fixed-length vector: HSV histogram, LBP histogram, and shape ratios."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("تصویر خالی یا ناخواناست.")
    image = cv2.resize(image_bgr, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    color_hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4],
                              [0, 180, 0, 256, 0, 256]).flatten()
    color_hist = color_hist.astype(np.float32)
    color_hist /= (color_hist.sum() + 1e-8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    texture_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 12), range=(0, 11))
    texture_hist = texture_hist.astype(np.float32)
    texture_hist /= (texture_hist.sum() + 1e-8)

    # Simple threshold-derived shape summary; quality depends on background contrast.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(mask) > 127:
        mask = cv2.bitwise_not(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shape = np.zeros(4, dtype=np.float32)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        x, y, w, h = cv2.boundingRect(contour)
        shape = np.array([
            w / (h + 1e-8),
            area / (w * h + 1e-8),
            4 * np.pi * area / (perimeter * perimeter + 1e-8),
            area / (gray.shape[0] * gray.shape[1])
        ], dtype=np.float32)
    return np.concatenate([color_hist, texture_hist, shape]).astype(np.float32)


def load_dataset(data_dir):
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"پوشه دیتاست پیدا نشد: {root}")
    features, labels, skipped = [], [], []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            img = cv2.imread(str(path))
            if img is None:
                skipped.append(str(path))
                continue
            features.append(extract_features(img))
            labels.append(class_dir.name)
    if not features:
        raise ValueError("تصویر قابل‌خواندن پیدا نشد؛ ساختار data/fruit/<class_name>/ را بررسی کنید.")
    return np.vstack(features), np.asarray(labels), skipped


def train_and_evaluate(X, y, test_size=0.25, random_state=21):
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("برای دسته‌بندی دست‌کم دو کلاس لازم است.")
    if counts.min() < 2:
        raise ValueError("هر کلاس باید دست‌کم دو تصویر داشته باشد.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=3.0, gamma="scale"))
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    pred = model.predict(X_test)
    predict_seconds = time.perf_counter() - start
    labels = sorted(np.unique(y).tolist())
    return {
        "model": model,
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
        "report": classification_report(y_test, pred, labels=labels, zero_division=0, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=labels),
        "labels": labels,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "test_count": len(y_test),
        "predictions": pred,
        "y_test": y_test,
    }


def predict_file(image_path, model):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"خواندن تصویر ممکن نیست: {image_path}")
    return str(model.predict(extract_features(image).reshape(1, -1))[0])
