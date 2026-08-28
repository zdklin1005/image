# Strict fruit-system regression report

## Verdict

Model A is now unrestricted and final detections use containment-aware,
centre-aligned deduplication. Labelled top-1 accuracy has returned to 80.7%,
and external class-presence recall increased substantially. The system is
still not reliably generalised to unseen Mango, Pear, and Peach images.

The fusion policy and Model A integration were updated; model weights were not
changed.

## Test coverage

- 140 labelled, single-fruit images sampled across Apple, Banana, Grape,
  Mango, and Orange.
- 13 original external/user images, including isolated fruits, mixed-fruit
  scenes, natural backgrounds, sliced fruit, and multiple instances.
- 78 external detector runs: original plus darkened, brightened, blurred,
  salt-and-pepper, and 15-degree rotated versions of every base image.
- 13 complete original-image system runs through preprocessing, detection,
  ROI segmentation, ripeness, and blemish analysis.
- 80 labelled ripeness-gate samples, of which 58 had a correct fruit detection
  and could be evaluated.
- Ten deterministic contract tests covering fusion, gates, box validation,
  ripeness drawing, resizing, and the valid-content mask.

## Labelled single-fruit results

| Expected fruit | Images | Current correct | Current accuracy |
|---|---:|---:|---:|
| Apple | 40 | 37 | 92.5% |
| Banana | 40 | 38 | 95.0% |
| Grape | 20 | 20 | 100.0% |
| Mango | 20 | 0 | 0.0% |
| Orange | 20 | 18 | 90.0% |
| **Overall** | **140** | **113** | **80.7%** |

The current system produced no usable detection on 3 images and 52 false-class
boxes. The 46 repeated same-class boxes include legitimate multiple instances,
especially grapes, so they must not be interpreted as 46 confirmed duplicates.

Compared with the restricted policy, unrestricted Model A gained 12 correct
images and lost none. Containment-aware final deduplication reduced the earlier
unrestricted run from 58 to 52 false-class boxes and from 49 to 46 repeated
same-class boxes while preserving neighbouring-fruit contract tests.

## External and robustness results

The 13 original external images contained 36 expected class-presence labels.
The system detected 18, giving 50.0% class-presence recall. Every original
image produced at least one detection. This is a presence metric, not bounding-box mAP, because
the external photos do not have box annotations.

| Variant | Expected classes found | Presence recall | No-detection images | False-class predictions |
|---|---:|---:|---:|---:|
| Original | 18/36 | 50.0% | 0 | 4 |
| Brightened | 16/36 | 44.4% | 1 | 4 |
| Darkened | 16/36 | 44.4% | 0 | 5 |
| Gaussian blur 9×9 | 17/36 | 47.2% | 0 | 4 |
| Salt-and-pepper 1% | 18/36 | 50.0% | 0 | 5 |
| Rotation 15° | 15/36 | 41.7% | 1 | 3 |

Notable original-image failures:

- The new internet Mango was labelled Apple, and the user three-Mango image
  produced three Apple boxes.
- The new Pear image was labelled Orange; the new Grape was detected correctly.
- The peach image was classified as Apple.
- The isolated pineapple was correctly detected.
- The exact mixed-fruit target scene now contains one Pineapple box rather than
  internal duplicate boxes. Pineapple remains absent from ripeness output.
- Its right-side green Mango is still labelled Apple, which requires improved
  model training rather than further deduplication.

## Preprocessing findings

- All preprocessing output-size and valid-content-mask contract tests passed.
- Strong exposure changes were often identified: 7 of 13 brightened images
  and 9 of 13 darkened images were marked `Review required`.
- None of the 13 deliberately Gaussian-blurred images was labelled `Blurry`.
  Blur scores fell by approximately 4% to 37%, but all remained well above the
  fixed threshold of 60. This threshold needs calibration against labelled
  acceptable/unacceptable blur examples.
- Salt-and-pepper noise did not lower expected-class recall relative to the
  already-low original result, which is consistent with the median filter
  helping against impulse noise. It did increase false predictions and
  duplicates, so it does not make detection robust by itself.
- Fruits already missed in clean originals remained missed after filtering.
  This is evidence against preprocessing being the primary external-recall
  bottleneck.

## Detection, ripeness, and downstream findings

- The Model C fruit-class gate passed and did not alter accuracy on the 58
  evaluable ripeness samples: both baseline and gated accuracy were 67.2%.
  It is safe, but this sample showed no measurable improvement.
- All ten detection contract tests passed. Pineapple/Watermelon remain
  detection-only, and their boxes do not appear in the ripeness overlay.
- The full pipeline completed without a top-level crash for all 13 originals.
  Four detections failed during blemish overlay creation.
  When its blemish mask contained no positive pixels, `cv2.addWeighted` returned
  `None`; assignment of that result raised a `TypeError`. This is a blemish
  visualisation defect, not a preprocessing or calibration failure.
- External ripeness accuracy cannot be calculated from these photos because
  they do not provide verified ripeness ground truth. A displayed confidence
  is not evidence that the ripeness label is correct.
- Segmentation cannot explain fruits that were never detected: in the current
  flow, ROI segmentation runs only after a detection box exists.

## Priority order

1. Improve/retrain the fruit detector, with Mango as the first critical class
   and external mixed-fruit scenes included in validation.
2. Add instance-labelled external validation data and report precision,
   recall, and mAP rather than only confidence values.
3. Calibrate the blur threshold using labelled blur severity, separately from
   detector accuracy.
4. Handle an empty blemish mask before creating the red overlay.
5. Re-evaluate ripeness on a ground-truth test set after the new Model E
   training finishes.

## Result files

- `unrestricted_vs_restricted_model_a.csv` and
  `unrestricted_vs_restricted_model_a_summary.json`: labelled detector comparison.
- `external_robustness.csv` and `external_robustness_summary.json`: original
  and perturbation results.
- `external_full_system.csv`: per-detection end-to-end results.
- `external_annotated/`: annotated original-image outputs.
- `robustness_variants/`: generated stress-test images.
- `detection_contracts.json`: deterministic rule-test results.
