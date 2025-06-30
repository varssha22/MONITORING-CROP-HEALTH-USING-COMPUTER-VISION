import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
import argparse

# Ensure source folder access
sys.path.append('./src')

from EfficientNet_b0_Architecture import EfficientNetB0
from EfficientNet_b0_CBAM_Architecture import EfficientNetB0_CBAM

# Preprocessing (same as training)
def preprocess(img, label):
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

# Load validation dataset
def load_val_data(val_dir="data/val", image_size=(224, 224), batch_size=32):
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=False
    ).map(preprocess).prefetch(tf.data.AUTOTUNE)
    
    return val_ds

# Model selector
def build_model(name):
    if name == 'efficientnetb0':
        return EfficientNetB0()
    elif name == 'efficientnetb0_cbam':
        return EfficientNetB0_CBAM()
    else:
        raise ValueError("Unsupported model. Use 'efficientnetb0' or 'efficientnetb0_cbam'.")

# Main Evaluation Logic
def main(args):
    val_ds = load_val_data()

    model = build_model(args.model)
    model.load_weights(args.weights_path)

    class_names = val_ds.class_names

    true_labels = []
    for _, labels in val_ds.unbatch():
        true_labels.append(tf.argmax(labels).numpy())
    true_labels = np.array(true_labels)

    pred_labels = []
    for images, _ in val_ds:
        preds = model.predict(images, verbose=0)
        pred_classes = np.argmax(preds, axis=1)
        pred_labels.extend(pred_classes)
    pred_labels = np.array(pred_labels)

    report = classification_report(true_labels, pred_labels, target_names=class_names)
    print(report)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained crop health model")
    parser.add_argument('--model', type=str, required=True, choices=['efficientnetb0', 'efficientnetb0_cbam'],
                        help='Model architecture to evaluate')
    parser.add_argument('--weights_path', type=str, required=True,
                        help='Path to the trained weights file (.h5)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save the classification report')

    args = parser.parse_args()
    main(args)
