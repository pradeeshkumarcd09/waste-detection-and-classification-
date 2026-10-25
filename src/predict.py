import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "waste_classifier.keras"
CLASS_NAMES_PATH = PROJECT_DIR / "results" / "class_names.json"


def main():
    parser = argparse.ArgumentParser(
        description="Classify a waste image using the trained CNN."
    )
    parser.add_argument("image", help="Path to an image file")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    model = tf.keras.models.load_model(MODEL_PATH)

    image = Image.open(image_path).convert("RGB")
    image = image.resize((128, 128))

    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    probabilities = model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    print(f"Predicted category: {class_names[predicted_index]}")
    print(f"Confidence: {probabilities[predicted_index] * 100:.2f}%")
    print("\nClass probabilities:")

    for index in np.argsort(probabilities)[::-1]:
        print(f"{class_names[index]}: {probabilities[index] * 100:.2f}%")


if __name__ == "__main__":
    main()
