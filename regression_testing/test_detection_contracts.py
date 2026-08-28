"""Fast contract checks for detection fusion, gating, and preprocessing."""

from pathlib import Path
import sys
import types

import numpy as np

# The compact test-only Python runtime does not ship Tk.  main.py only needs
# these names for its interactive file picker, which this headless test never
# calls, so provide a minimal import stub before loading the drawing function.
if "tkinter" not in sys.modules:
    tkinter_stub = types.ModuleType("tkinter")
    tkinter_stub.Tk = object
    tkinter_stub.filedialog = object()
    sys.modules["tkinter"] = tkinter_stub

from fruit_ripeness_object_detection.fruit_detection import (
    assess_detection_quality,
    fuse_detections,
    suppress_detections_contained_by_extension_fruits,
)
from main import draw_ripeness_results
from preprocessing import preprocess_fruit_image


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def raw(model, fruit, confidence, box, ripeness=None):
    return {
        "model": model,
        "fruit_type": fruit,
        "confidence": confidence,
        "bounding_box": box,
        "ripeness": ripeness,
    }


def run_test(name, function, results):
    try:
        function()
        results.append({"test": name, "status": "PASS", "error": ""})
    except Exception as error:
        results.append({
            "test": name,
            "status": "FAIL",
            "error": f"{type(error).__name__}: {error}",
        })


def main():
    results = []

    def pineapple_is_detection_only():
        fused = fuse_detections(
            [raw("A", "Pineapple", 0.9, (20, 20, 200, 220))], [], []
        )
        assert len(fused) == 1
        assert fused[0]["fruit_type"] == "Pineapple"
        assert fused[0]["ripeness_supported"] is False
        assert fused[0]["support_level"] == "Detection only"

    def contained_false_box_is_removed():
        detections = [
            {"fruit_type": "Pineapple", "bounding_box": (10, 10, 300, 300)},
            {"fruit_type": "Apple", "bounding_box": (80, 80, 140, 140)},
        ]
        filtered = suppress_detections_contained_by_extension_fruits(detections)
        assert [item["fruit_type"] for item in filtered] == ["Pineapple"]

    def neighbouring_box_survives():
        detections = [
            {"fruit_type": "Pineapple", "bounding_box": (100, 20, 300, 300)},
            {"fruit_type": "Apple", "bounding_box": (20, 180, 160, 340)},
        ]
        filtered = suppress_detections_contained_by_extension_fruits(detections)
        assert len(filtered) == 2

    def adjacent_fruits_do_not_merge():
        fused = fuse_detections(
            [],
            [raw("C", "Apple", 0.8, (10, 10, 160, 200), "Ripe")],
            [raw("D", "Mango", 0.8, (120, 20, 280, 210))],
        )
        assert len(fused) == 2

    def contained_same_fruit_boxes_merge():
        fused = fuse_detections(
            [],
            [raw("C", "Apple", 0.8, (10, 10, 250, 250), "Ripe")],
            [raw("D", "Apple", 0.7, (60, 60, 180, 180))],
        )
        assert len(fused) == 1

    def neighbouring_same_class_fruits_remain_separate():
        fused = fuse_detections(
            [],
            [raw("C", "Apple", 0.8, (10, 10, 160, 200), "Ripe")],
            [raw("D", "Apple", 0.8, (120, 20, 280, 210))],
        )
        assert len(fused) == 2

    def model_c_ripeness_mismatch_is_blocked():
        fused = fuse_detections(
            [],
            [raw("C", "Orange", 0.4, (10, 10, 200, 200), "Ripe")],
            [raw("D", "Apple", 0.9, (10, 10, 200, 200))],
            cross_class_iou_threshold=0.60,
        )
        assert len(fused) == 1
        assert fused[0]["fruit_type"] == "Apple"
        assert fused[0]["model_c_matches_final"] is False
        assert fused[0]["model_c_ripeness"] is None

    def malformed_box_is_rejected():
        detection = {
            "fruit_type": "Apple",
            "confidence": 0.9,
            "bounding_box": (30, 30, 30, 100),
            "models": ["D"],
            "class_disagreement": False,
        }
        assessed = assess_detection_quality(
            [detection], (640, 640, 3), retain_rejected=True
        )
        assert assessed[0]["reliability_status"] == "Rejected"

    def ripeness_drawing_omits_detection_only_box():
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        result = {
            "bounding_box": (20, 20, 200, 200),
            "ripeness_supported": False,
            "final_ripeness": None,
            "final_confidence": None,
        }
        drawn = draw_ripeness_results(image, [result])
        assert np.array_equal(drawn, image)

    def preprocessing_contract_is_consistent():
        output = preprocess_fruit_image(PROJECT_ROOT / "img_5874-1.png")
        for key in (
            "analysis_image",
            "classification_image",
            "original_image",
            "display_image",
            "valid_content_mask",
        ):
            assert output[key].shape[:2] == (640, 640)
        assert output["valid_content_mask"].ndim == 2
        assert set(np.unique(output["valid_content_mask"])).issubset({0, 255})
        assert output["valid_content_mask"].sum() > 0

    checks = (
        ("pineapple detection-only gate", pineapple_is_detection_only),
        ("contained false box suppression", contained_false_box_is_removed),
        ("neighbouring fruit preservation", neighbouring_box_survives),
        ("adjacent fruit separation", adjacent_fruits_do_not_merge),
        ("contained same-fruit box deduplication", contained_same_fruit_boxes_merge),
        ("neighbouring same-class fruit preservation", neighbouring_same_class_fruits_remain_separate),
        ("Model C ripeness class gate", model_c_ripeness_mismatch_is_blocked),
        ("malformed box rejection", malformed_box_is_rejected),
        ("ripeness output hides detection-only fruit", ripeness_drawing_omits_detection_only_box),
        ("preprocessing output and content-mask contract", preprocessing_contract_is_consistent),
    )
    for name, function in checks:
        run_test(name, function, results)

    output = PROJECT_ROOT / "strict_regression_results/detection_contracts.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    import json
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for result in results:
        print(f"{result['status']:4} | {result['test']} {result['error']}")
    failed = [result for result in results if result["status"] == "FAIL"]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
