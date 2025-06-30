import tensorflow as tf
import numpy as np
import cv2
import os

# Get nth last Conv2D layer name
def get_nth_last_conv_layer_name(model, n=1):
    conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    if len(conv_layers) < n:
        raise ValueError(f"Model has only {len(conv_layers)} Conv2D layers, can't get {n}-th last.")
    return conv_layers[-n].name


# Grad-CAM++ implementation
def gradcam_plus_plus(img, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape(persistent=True) as tape1:
        with tf.GradientTape(persistent=True) as tape2:
            with tf.GradientTape() as tape3:
                conv_outputs, predictions = grad_model(img)
                pred_index = tf.argmax(predictions[0])
                class_channel = predictions[:, pred_index]

            grads = tape3.gradient(class_channel, conv_outputs)
        first_derivative = grads
        second_derivative = tape2.gradient(grads, conv_outputs)
    third_derivative = tape1.gradient(second_derivative, conv_outputs)

    conv_outputs = conv_outputs[0].numpy()
    first_derivative = first_derivative[0].numpy()
    second_derivative = second_derivative[0].numpy()
    third_derivative = third_derivative[0].numpy()

    global_sum = np.sum(conv_outputs, axis=(0, 1))
    alpha_num = second_derivative
    alpha_denom = 2 * second_derivative + third_derivative * global_sum[np.newaxis, np.newaxis, :]
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1e-10)

    alphas = alpha_num / alpha_denom
    weights = np.maximum(first_derivative, 0.0)
    deep_linearization_weights = np.sum(alphas * weights, axis=(0, 1))

    heatmap = np.sum(deep_linearization_weights * conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    return heatmap


# Function to overlay Grad-CAM mask with outline
def draw_gradcam_fill_and_outline(heatmap, original_img, threshold_ratio=0.5,
                                  fill_color=(255, 150, 150), outline_color=(0, 255, 0),
                                  fill_alpha=0.4, outline_thickness=1):
    if original_img.max() <= 1.0:
        img_uint8 = (original_img * 255).astype(np.uint8)
    else:
        img_uint8 = original_img.copy()

    heatmap_resized = cv2.resize(heatmap, (img_uint8.shape[1], img_uint8.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    threshold = int(threshold_ratio * 255)
    _, binary_mask = cv2.threshold(heatmap_resized, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

    overlay = np.full_like(img_uint8, fill_color, dtype=np.uint8)
    mask_3ch = np.stack([cleaned_mask]*3, axis=-1) // 255
    filled_img = np.where(mask_3ch, cv2.addWeighted(img_uint8, 1 - fill_alpha, overlay, fill_alpha, 0), img_uint8)

    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outlined = cv2.drawContours(filled_img.copy(), contours, -1, outline_color, outline_thickness)

    return (outlined / 255.0) if original_img.max() <= 1.0 else outlined
