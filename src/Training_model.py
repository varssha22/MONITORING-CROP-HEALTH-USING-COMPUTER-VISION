import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import argparse
import os
import sys
from tensorflow.keras import layers, models

sys.path.append('./src')
from EfficientNet_b0_CBAM_Architecture import EfficientNetB0_CBAM
from EfficientNet_b0_Architecture import EfficientNetB0

# Normalize images
def preprocess(img, label):
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

# Data loading function
def get_data(train_dir="data/train", val_dir="data/val", image_size=(224, 224), batch_size=32):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=True
    ).map(preprocess).repeat().prefetch(tf.data.AUTOTUNE)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=False
    ).map(preprocess).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds

# Model selector
def build_model(name):
    if name == 'efficientnetb0':
        model = EfficientNetB0()
    elif name == 'efficientnetb0_cbam':
        model = EfficientNetB0_CBAM()
    else:
        raise ValueError("Unsupported model. Use 'efficientnetb0' or 'efficientnetb0_cbam'.")
    return model

# Optional: Plot function placeholder
def plot_training(history, save_dir):
    import matplotlib.pyplot as plt
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Training & Validation Metrics')
    plt.savefig(os.path.join(save_dir, "training_plot.png"))
    plt.close()

# Main
def main(args):
    train_ds, val_ds = get_data()

    model = build_model(args.model)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    checkpoint_cb = ModelCheckpoint('models/best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')
    csv_logger = CSVLogger('training_log.csv', append=True)

    model.summary()
    # Dynamically calculate total number of images
    train_img_count = sum([len(files) for r, d, files in os.walk("data/train")])
    val_img_count = sum([len(files) for r, d, files in os.walk("data/val")])

    # Steps calculation
    batch_size = 32
    steps_per_epoch = math.ceil(train_img_count / batch_size)
    validation_steps = math.ceil(val_img_count / batch_size)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,  # Adjust based on dataset size
        validation_steps=validation_steps,  # Adjust based on dataset size
        callbacks=[checkpoint_cb, csv_logger]
    )

    os.makedirs(args.save_dir, exist_ok=True)
    model.save(os.path.join(args.save_dir, f"{args.model}_crop_health_model.h5"))
    plot_training(history, args.save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train crop health classification model")
    parser.add_argument('--model', type=str, required=True, choices=['efficientnetb0', 'efficientnetb0_cbam'],
                        help='Model architecture: efficientnetb0 or efficientnetb0_cbam')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save model and plots')

    args = parser.parse_args()
    main(args)
