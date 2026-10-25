# Vision-Based Waste Detection and Classification

A deep-learning project that classifies waste images into six categories
using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

## Waste Categories
- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

## Dataset
This project uses the TrashNet dataset.
The extracted dataset used for this experiment contains 2,527 images.

Dataset source: https://github.com/garythung/trashnet

The images were resized to 128 x 128 pixels and pixel values were
normalized to the range 0-1.

## Model
- Framework: TensorFlow / Keras
- Architecture: CNN
- Input shape: 128 x 128 x 3
- Output classes: 6
- Optimizer: Adam
- Loss: Sparse categorical cross-entropy
- Training: Early stopping and model checkpointing

## Dataset Split
- Training: 1,768 images
- Validation: 379 images
- Testing: 380 images

## Test Results
- Test accuracy: 63.16%
- Macro F1-score: 61.76%
- Weighted F1-score: 62.71%

See `results/classification_report.json` for class-level metrics and
`results/confusion_matrix.png` for the confusion matrix.

These are the results from this particular training run. Performance
may vary with random initialization, software versions, and hardware.

## Project Structure
```text
waste-detection-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── predict.py
├── models/
│   └── waste_classifier.keras
├── results/
│   ├── class_names.json
│   ├── classification_report.json
│   ├── test_metrics.json
│   └── confusion_matrix.png
└── screenshots/
```

## Setup
Use Python 3.10 or 3.11 in a compatible TensorFlow environment.

Install dependencies:
```bash
pip install -r requirements.txt
```

## Run Prediction
Place an image file on your computer, then run:
```bash
python src/predict.py path/to/waste_image.jpg
```

The script prints the predicted category and class probabilities.

## Limitations
This is a baseline model trained on a relatively small dataset.
The observed test accuracy was 63.16%, and some categories are frequently
confused. Results on real-world images may differ from results on TrashNet.
This project is an educational prototype, not a production waste-sorting system.

## License
Check the dataset repository's license and terms before redistributing
the dataset or using it commercially. The model and code should be
used subject to their applicable licenses.
