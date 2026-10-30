from pathlib import Path
import csv

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

SPLITS = ["train", "validation", "test"]

CLASSES = [
    "Apple",
    "Banana",
    "Pomegranate",
    "Watermelon"
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def count_images(folder):

    if not folder.exists():
        return 0

    return sum(
        1
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main():

    rows = []

    print("\nDATASET SPLIT REPORT")
    print("=" * 60)

    for split in SPLITS:

        split_total = 0

        print(f"\n{split.upper()}")

        for class_name in CLASSES:

            class_dir = DATA_DIR / split / class_name

            count = count_images(class_dir)
            split_total += count

            print(f"{class_name:15s}: {count}")

            rows.append({
                "split": split,
                "class": class_name,
                "image_count": count
            })

        print(f"Total {split}: {split_total}")

    report_path = BASE_DIR / "dataset_split_report.csv"

    with open(
        report_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split",
                "class",
                "image_count"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()