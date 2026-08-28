# External Multi-Fruit System Evaluation

## Method

The current integrated pipeline was evaluated on four externally sourced,
previously unseen multi-fruit photographs. The runtime configuration was kept
unchanged: 640 x 640 letterbox standardisation, median 5 x 5 filtering,
bilateral filtering, three YOLO detectors at a 0.30 confidence threshold,
spatial detection fusion, ROI Otsu segmentation, ripeness fusion and
fruit-specific blemish analysis. No model was retrained or modified.

The source descriptions provide fruit classes but not bounding-box
annotations. Consequently, the class-presence figures below are a manual
external diagnostic, not mAP or a replacement for a labelled detection test.

## Detection results

| Image | Source-described supported classes present | Final predicted classes | Main outcome |
|---|---|---|---|
| `01_fruit_bowl.jpg` | Apple, Banana, Orange, Pear | Apple x2, Mango x1 | One visible apple was correctly identified; the pomegranate was labelled Apple, another region was labelled Mango, and Banana/Orange/Pear were missed. |
| `02_bowl_of_fruit.jpg` | Apple, Orange | Orange x5, Mango x1 | All five visible oranges were localized as Orange. One apple was localized but labelled Mango; the remaining apples were missed. |
| `03_culinary_fruits.jpg` | Apple, Banana, Grape, Mango, Melon, Orange, Pear | Apple x2, Banana x2, Pineapple x1, Watermelon x1 | Large foreground objects were favoured. Pear, Grape, Mango, Melon and Orange were missed. Pineapple and Watermelon were returned even though they are outside the declared eight-class product scope. |
| `04_fruits_in_basket.jpg` | Apple, Banana, Grape, Orange | Apple x4, Banana x1 | All four apples and the banana bunch were detected. Grapes and Orange were missed. |

Across the four images, 6 of 17 supported image-class occurrences were
recovered, giving class-presence recall of 35.3%. Six of ten unique class
outputs matched a source-described supported class in the corresponding image,
giving unfiltered class-presence precision of 60.0%. If unsupported Pineapple
and Watermelon outputs are excluded, supported-only precision is 75.0%.

These figures measure whether a class was present at least once per image; they
do not measure box-level precision, instance counts or mAP.

## Preprocessing diagnostic

Resize-only and current processed inputs produced complementary results. The
resize-only variant generally retained higher confidence for large fruit. For
example, five Orange detections in `02_bowl_of_fruit.jpg` ranged from 75.5% to
94.2% with resize-only input, compared with 30.3% to 77.1% after the current
median-5 and bilateral filtering. However, processed input recovered Banana in
`04_fruits_in_basket.jpg` much more strongly (80.5% versus 38.3%).

Therefore, this small test does not justify globally replacing processed input
with raw or resize-only input. It shows that one fixed preprocessing strength
does not benefit every fruit, model and scene equally.

## Ripeness and blemish observations

Eighteen of twenty detections were classified as Ripe and two as Unripe. The
external sources do not provide verified ripeness ground truth, so a ripeness
accuracy percentage cannot be calculated. Several incorrect fruit detections
still received high ripeness confidence, demonstrating error propagation from
detection into classification.

Blemish results were plausible for the isolated oranges in
`02_bowl_of_fruit.jpg` (approximately 1.6% to 9.2%) but became implausibly high
for several crowded or incorrect ROIs, including approximately 35% to 53% in
`01_fruit_bowl.jpg` and up to approximately 51% in
`03_culinary_fruits.jpg`. These values should not be treated as verified defect
measurements because no blemish masks or ground-truth percentages are supplied.

## Recommended no-retraining fixes

1. Enforce the declared eight-class output set and return `Unsupported` rather
   than Pineapple, Watermelon or unknown model classes.
2. Use Model C ripeness only when Model C's fruit class agrees with the final
   fused fruit class.
3. Mark detections for manual review when confidence is below 50%, only one
   model supports the box, or the models disagree on the fruit class.
4. Add an optional tiled or higher-resolution detection pass for crowded
   images so small fruit are not lost when the complete scene is reduced to
   640 x 640.
5. Do not report blemish percentage when the segmentation mask fails quality
   checks, such as excessive ROI coverage, border contact or very small area.
6. Present multi-fruit detection as supported, while documenting that reliable
   ripeness and blemish measurement requires a sufficiently isolated ROI for
   each fruit.

## Generated outputs

- `outputs/external_detection_results.csv`: one row per final detection.
- `outputs/external_detection_summary.json`: per-image pipeline summary.
- `outputs/*_detections.jpg`: annotated detector outputs.
