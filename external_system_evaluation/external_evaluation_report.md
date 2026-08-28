# External Multi-Fruit System Evaluation

## Method

The current integrated pipeline was evaluated on four externally sourced,
previously unseen multi-fruit photographs. The runtime configuration was kept
unchanged at model level: 640 x 640 letterbox standardisation, median 5 x 5
filtering, bilateral filtering, three YOLO detectors, spatial detection
fusion, ROI Otsu segmentation, ripeness fusion and fruit-specific blemish
analysis. The inference pipeline now applies per-class confidence thresholds,
detection reliability checks and supported-class gating. No model was
retrained or modified.

The source descriptions provide fruit classes but not bounding-box
annotations. Consequently, the class-presence figures below are a manual
external diagnostic, not mAP or a replacement for a labelled detection test.

## Detection results

| Image | Source-described supported classes present | Final predicted classes | Main outcome |
|---|---|---|---|
| `01_fruit_bowl.jpg` | Apple, Banana, Orange, Pear | Apple x2 | One visible apple was correctly identified; the pomegranate was still labelled Apple. The previous false Mango output was removed. |
| `02_bowl_of_fruit.jpg` | Apple, Orange | Orange x5, Apple x1 | All five visible oranges were localized as Orange. The region previously labelled Mango is now labelled Apple. |
| `03_culinary_fruits.jpg` | Apple, Banana, Grape, Mango, Melon, Orange, Pear | Apple x2, Banana x2, Unsupported x2 | Large foreground objects were favoured. Pineapple and Watermelon are now displayed as Unsupported and blocked from downstream analysis. |
| `04_fruits_in_basket.jpg` | Apple, Banana, Grape, Orange | Apple x4, Banana x1 | All four apples and the banana bunch were detected. Grapes and Orange were missed. |

Across the four images, 7 of 17 supported image-class occurrences were
recovered, giving class-presence recall of 41.2%, compared with 35.3% before
per-class thresholding. Every supported class name returned by the updated
pipeline was present somewhere in its corresponding image. This does not prove
box-level precision: for example, the pomegranate in image 1 is still labelled
Apple.

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

The external sources do not provide verified ripeness ground truth, so a
ripeness accuracy percentage cannot be calculated. Unsupported detections are
now blocked before ripeness analysis, and Model C evidence is ignored whenever
its fruit class disagrees with the fused fruit class.

Blemish percentages remain dependent on the segmentation masks supplied by the
existing segmentation module. The external photographs do not include blemish
ground truth, so these values must not be interpreted as verified defect
measurements.

## Current controls and remaining recommendations

1. Enforce the declared eight-class output set and return `Unsupported` rather
   than Pineapple, Watermelon or unknown model classes.
2. Use Model C ripeness only when Model C's fruit class agrees with the final
   fused fruit class.
3. Apply per-class confidence thresholds and mark single-model, disagreement
   and partially padded boxes for review.
4. Keep tiled preparation available as an experimental option; it is not
   enabled automatically because it also introduced false positives.
5. A future segmentation-owned validation step could withhold blemish
   percentage when that module determines its mask is unsuitable.
6. Present multi-fruit detection as supported, while documenting that reliable
   ripeness and blemish measurement requires a sufficiently isolated ROI for
   each fruit.

## Generated outputs

- `outputs/external_detection_results.csv`: one row per final detection.
- `outputs/external_detection_summary.json`: per-image pipeline summary.
- `outputs/*_detections.jpg`: annotated detector outputs.
