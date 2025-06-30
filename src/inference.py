import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Ensure src folder access
sys.path.append('./src')

from EfficientNet_b0_Architecture import EfficientNetB0
from EfficientNet_b0_CBAM_Architecture import EfficientNetB0_CBAM
from gradcam import gradCAM_plus_plus, draw_gradcam_fill_and_outline, get_nth_last_conv_layer_name

# Test images
image_names = [
    'AppleCedarRust1', 'AppleCedarRust2', 'AppleCedarRust3', 'AppleCedarRust4',
    'AppleScab1', 'AppleScab2', 'AppleScab3',
    'CornCommonRust1', 'CornCommonRust2', 'CornCommonRust3',
    'PotatoEarlyBlight1', 'PotatoEarlyBlight2', 'PotatoEarlyBlight3',
    'PotatoEarlyBlight4', 'PotatoEarlyBlight5', 'PotatoHealthy1', 'PotatoHealthy2',
    'TomatoEarlyBlight1', 'TomatoEarlyBlight2', 'TomatoEarlyBlight3', 'TomatoEarlyBlight4', 'TomatoEarlyBlight5',
    'TomatoEarlyBlight6', 'TomatoHealthy1', 'TomatoHealthy2', 'TomatoHealthy3', 'TomatoHealthy4',
    'TomatoYellowCurlVirus1', 'TomatoYellowCurlVirus2', 'TomatoYellowCurlVirus3',
    'TomatoYellowCurlVirus4', 'TomatoYellowCurlVirus5', 'TomatoYellowCurlVirus6'
]

image_paths = [f"./data/test_images/{img}.JPG" for img in image_names]

# Image settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

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
    # Validation dataset for class names
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "data/val",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False
    )
    class_names = val_ds.class_names

    model = build_model(args.model)
    model.load_weights(args.weights_path)

    last_conv_layer_name = get_nth_last_conv_layer_name(model, n=43)  # Adjust as per your architecture

    output_dir = "./gradcam_outputs"
    os.makedirs(output_dir, exist_ok=True)

    for i, img_path in enumerate(image_paths):
        true_class = image_names[i]

        # Load and preprocess image
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Prediction
        preds = model.predict(img_array, verbose=0)
        pred_idx = np.argmax(preds)
        pred_class = class_names[pred_idx]

        print(f"[{i+1}] True Class: {true_class} | Predicted Class: {pred_class}")

        # Grad-CAM++
        heatmap = gradCAM_plus_plus(img_array, model, last_conv_layer_name)

        # Threshold
        percentile = 85
        threshold_value = np.percentile(heatmap, percentile)
        threshold_ratio = threshold_value / (np.max(heatmap) + 1e-8)

        outlined_img = draw_gradcam_fill_and_outline(heatmap, img_array[0], threshold_ratio=threshold_ratio)

        # Plot and save
        plt.figure(figsize=(12, 5))
        plt.suptitle(f"True: {true_class} | Predicted: {pred_class}", fontsize=14)

        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(img_array[0])
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Grad-CAM++ Heatmap")
        plt.imshow(heatmap, cmap='jet')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Outlined Grad-CAM++ Region")
        plt.imshow(outlined_img)
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'gradcam_output_{i+1}.jpg'))
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM++ Inference for Crop Health Model")
    parser.add_argument('--model', type=str, required=True, choices=['efficientnetb0', 'efficientnetb0_cbam'],
                        help='Model architecture to use')
    parser.add_argument('--weights_path', type=str, required=True, help='Path to the trained model weights (.h5)')

    args = parser.parse_args()
    main(args)
