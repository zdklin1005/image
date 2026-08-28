from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


# ============================================================
# MODEL PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_B_PATH = (
    PROJECT_ROOT
    / "models"
    / "model_b_ripeness.keras"
)

MODEL_E_PATH = (
    PROJECT_ROOT
    / "models"
    / "model_e_ripeness.keras"
)


# ============================================================
# LOAD MODELS
# ============================================================

model_b = tf.keras.models.load_model(
    str(MODEL_B_PATH)
)

model_e = tf.keras.models.load_model(
    str(MODEL_E_PATH)
)


# ============================================================
# CLASS NAMES
# ============================================================

# Model B:
# 2-class ripeness model
MODEL_B_CLASSES = [
    "Ripe",
    "Unripe"
]


# Model E:
# Based on your Colab training folder order
MODEL_E_CLASSES = [
    "Overripe",
    "Ripe",
    "Rotten",
    "Unripe"
]


# ============================================================
# PREPARE IMAGE
# ============================================================

def prepare_ripeness_image(
    image,
    image_size=(224, 224)
):
    """
    Resize fruit ROI for MobileNetV2 models.
    """

    if image is None:
        raise ValueError(
            "Ripeness input image is None."
        )

    if image.size == 0:
        raise ValueError(
            "Ripeness input image is empty."
        )

    resized = cv2.resize(
        image,
        image_size,
        interpolation=cv2.INTER_AREA
    )

    # OpenCV uses BGR.
    # TensorFlow training images are RGB.
    rgb_image = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )

    image_array = np.asarray(
        rgb_image,
        dtype=np.float32
    )

    # Add batch dimension:
    # (224,224,3)
    # ->
    # (1,224,224,3)
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# MODEL B
# ============================================================

def classify_with_model_b(
    fruit_image
):
    """
    Model B predicts:
        Ripe
        Unripe

    Model B is coarse supporting evidence.
    """

    image_array = prepare_ripeness_image(
        fruit_image
    )

    predictions = model_b.predict(
        image_array,
        verbose=0
    )[0]

    class_index = int(
        np.argmax(predictions)
    )

    confidence = float(
        predictions[class_index]
    )

    ripeness = MODEL_B_CLASSES[
        class_index
    ]

    return {
        "model": "B",
        "ripeness": ripeness,
        "confidence": confidence,
        "probabilities": {
            class_name: float(
                predictions[index]
            )
            for index, class_name
            in enumerate(
                MODEL_B_CLASSES
            )
        }
    }


# ============================================================
# MODEL E
# ============================================================

def classify_with_model_e(
    fruit_image
):
    """
    Model E predicts:
        Overripe
        Ripe
        Rotten
        Unripe
    """

    image_array = prepare_ripeness_image(
        fruit_image
    )

    predictions = model_e.predict(
        image_array,
        verbose=0
    )[0]

    class_index = int(
        np.argmax(predictions)
    )

    confidence = float(
        predictions[class_index]
    )

    ripeness = MODEL_E_CLASSES[
        class_index
    ]

    return {
        "model": "E",
        "ripeness": ripeness,
        "confidence": confidence,
        "probabilities": {
            class_name: float(
                predictions[index]
            )
            for index, class_name
            in enumerate(
                MODEL_E_CLASSES
            )
        }
    }


# ============================================================
# NORMALISE RIPENESS
# ============================================================

def normalise_ripeness(
    ripeness
):
    if ripeness is None:
        return None

    value = str(
        ripeness
    ).strip().lower()

    mapping = {
        "overripe": "Overripe",
        "ripe": "Ripe",
        "rotten": "Rotten",
        "unripe": "Unripe"
    }

    return mapping.get(
        value
    )


# ============================================================
# FUSE MODEL B + MODEL C + MODEL E
# ============================================================

def fuse_ripeness(
    result_b,
    model_c_ripeness,
    model_c_confidence,
    result_e
):
    """
    Fuse Model B, Model C and Model E.

    Model E:
        Main 4-class classifier.

    Model C:
        Supporting 4-class evidence.

    Model B:
        Coarse supporting evidence:
        Ripe / Unripe only.
    """

    final_classes = [
        "Overripe",
        "Ripe",
        "Rotten",
        "Unripe"
    ]

    scores = {
        class_name: 0.0
        for class_name
        in final_classes
    }

    # ========================================================
    # MODEL E
    # 60% WEIGHT
    # ========================================================

    for class_name in final_classes:

        probability = (
            result_e[
                "probabilities"
            ].get(
                class_name,
                0.0
            )
        )

        scores[class_name] += (
            0.60
            * probability
        )

    # ========================================================
    # MODEL C
    # 25% SUPPORT
    # ========================================================

    c_ripeness = normalise_ripeness(
        model_c_ripeness
    )

    if (
        c_ripeness is not None
        and model_c_confidence
        is not None
    ):

        scores[c_ripeness] += (
            0.25
            * float(
                model_c_confidence
            )
        )

    # ========================================================
    # MODEL B
    # 15% COARSE SUPPORT
    # ========================================================

    b_ripeness = result_b[
        "ripeness"
    ]

    b_confidence = result_b[
        "confidence"
    ]

    if b_ripeness == "Unripe":

        scores["Unripe"] += (
            0.15
            * b_confidence
        )

    elif b_ripeness == "Ripe":

        # Model B cannot distinguish
        # Ripe from Overripe.
        scores["Ripe"] += (
            0.075
            * b_confidence
        )

        scores["Overripe"] += (
            0.075
            * b_confidence
        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    final_ripeness = max(
        scores,
        key=scores.get
    )

    total_score = sum(
        scores.values()
    )

    if total_score > 0:

        final_confidence = (
            scores[
                final_ripeness
            ]
            / total_score
        )

    else:

        final_confidence = 0.0

    return {
        "ripeness": final_ripeness,
        "confidence": float(
            final_confidence
        ),
        "scores": scores
    }
