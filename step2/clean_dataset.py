from pathlib import Path
from PIL import Image, UnidentifiedImageError
import csv

# مسیر اصلی پروژه
BASE_DIR = Path(__file__).resolve().parent

# مسیر دیتاست آماده‌شده
DATA_DIR = BASE_DIR / "data" / "processed"

# کلاس‌های پروژه
CLASSES = [
    "Apple",
    "Banana",
    "Pomegranate",
    "Watermelon"
]

# پسوندهای رایج تصاویر
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

# حداقل ابعاد پیشنهادی برای علامت‌گذاری
MIN_WIDTH = 32
MIN_HEIGHT = 32


def inspect_image(image_path):
    """
    بررسی اعتبار یک فایل تصویری.
    فایل حذف یا تغییر داده نمی‌شود.
    """

    try:
        with Image.open(image_path) as image:
            image.verify()

        # باز کردن مجدد برای خواندن ابعاد
        with Image.open(image_path) as image:
            width, height = image.size

        if width < MIN_WIDTH or height < MIN_HEIGHT:
            return {
                "status": "too_small",
                "width": width,
                "height": height
            }

        return {
            "status": "valid",
            "width": width,
            "height": height
        }

    except (
        UnidentifiedImageError,
        OSError,
        ValueError
    ):
        return {
            "status": "corrupt",
            "width": "",
            "height": ""
        }


def main():

    if not DATA_DIR.exists():
        print(f"Dataset directory not found: {DATA_DIR}")
        return

    report = []

    for split in ["train", "validation", "test"]:

        split_dir = DATA_DIR / split

        if not split_dir.exists():
            print(f"Missing split: {split}")
            continue

        print(f"\nChecking {split.upper()} dataset")

        for class_name in CLASSES:

            class_dir = split_dir / class_name

            if not class_dir.exists():
                print(f"Missing class directory: {class_dir}")
                continue

            valid_count = 0
            corrupt_count = 0
            small_count = 0
            unsupported_count = 0

            for file_path in class_dir.iterdir():

                if not file_path.is_file():
                    continue

                # فایل‌های دارای پسوند ناشناخته
                if file_path.suffix.lower() not in IMAGE_EXTENSIONS:

                    unsupported_count += 1

                    report.append({
                        "split": split,
                        "class": class_name,
                        "file": str(file_path.relative_to(BASE_DIR)),
                        "status": "unsupported_extension",
                        "width": "",
                        "height": ""
                    })

                    continue

                result = inspect_image(file_path)

                report.append({
                    "split": split,
                    "class": class_name,
                    "file": str(file_path.relative_to(BASE_DIR)),
                    "status": result["status"],
                    "width": result["width"],
                    "height": result["height"]
                })

                if result["status"] == "valid":
                    valid_count += 1

                elif result["status"] == "corrupt":
                    corrupt_count += 1

                elif result["status"] == "too_small":
                    small_count += 1

            print(f"\nClass: {class_name}")
            print(f"Valid images: {valid_count}")
            print(f"Corrupt images: {corrupt_count}")
            print(f"Very small images: {small_count}")
            print(f"Unsupported files: {unsupported_count}")

    # ذخیره گزارش
    report_path = BASE_DIR / "data_cleaning_report.csv"

    with open(
        report_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as csv_file:

        fieldnames = [
            "split",
            "class",
            "file",
            "status",
            "width",
            "height"
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(report)

    print("\nCleaning inspection completed.")
    print(f"Report saved at: {report_path}")
    print("No files were deleted or modified.")


if __name__ == "__main__":
    main()