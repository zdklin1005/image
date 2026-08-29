from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


CURRENT_FOLDER = Path(__file__).resolve().parent

CSV_PATH = (
    CURRENT_FOLDER
    / "segmentation_iou_results.csv"
)

OUTPUT_PATH = (
    CURRENT_FOLDER
    / "segmentation_iou_by_fruit.png"
)


# Read evaluation results
df = pd.read_csv(CSV_PATH)

# Extract fruit type from filenames
# Example:
# apple_01.png -> Apple
df["fruit"] = (
    df["image"]
    .str.split("_")
    .str[0]
    .str.capitalize()
)

# Group by fruit type
summary = (
    df.groupby("fruit")
    .agg(
        Images=("image", "count"),
        Mean_IoU=("iou", "mean")
    )
    .reset_index()
)

# Optional order
fruit_order = [
    "Apple",
    "Banana",
    "Grape",
    "Mango",
    "Orange",
    "Peach",
    "Pear",
    "Watermelon"
]

summary["fruit"] = pd.Categorical(
    summary["fruit"],
    categories=fruit_order,
    ordered=True
)

summary = summary.sort_values("fruit")


# Labels such as Apple (n=7)
labels = [
    f"{row.fruit}\n(n={row.Images})"
    for row in summary.itertuples()
]


# Create chart
plt.figure(figsize=(10, 6))

bars = plt.bar(
    labels,
    summary["Mean_IoU"]
)

# Objective threshold
plt.axhline(
    y=0.70,
    linestyle="--",
    label="Target Mean IoU = 0.70"
)

# Add mean IoU above each bar
for bar, value in zip(
    bars,
    summary["Mean_IoU"]
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.015,
        f"{value:.4f}",
        ha="center",
        va="bottom"
    )

plt.xlabel("Fruit Type")
plt.ylabel("Mean Intersection over Union (IoU)")

plt.title(
    "Mean Segmentation IoU by Fruit Type"
)

plt.ylim(0, 1.0)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    f"Chart saved to: {OUTPUT_PATH}"
)