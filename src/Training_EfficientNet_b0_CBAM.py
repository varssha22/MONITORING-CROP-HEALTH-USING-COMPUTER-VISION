import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.metrics import classification_report
import sys

sys.path.append('./src')
from EfficientNet_CBAM_Architecture import EfficientNetB0_custom  

# Load model
model_e_cbam = EfficientNetB0_custom()

# Directories
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"  # Add your validation directory here
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Normalize images
def preprocess(img, label):
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

# Training Dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True
).map(preprocess).repeat().prefetch(tf.data.AUTOTUNE)

# Validation Dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False
).map(preprocess).prefetch(tf.data.AUTOTUNE)

# Calculate steps
train_image_count = len(list(tf.keras.preprocessing.image_dataset_from_directory(TRAIN_DIR, image_size=IMAGE_SIZE)))
val_image_count = len(list(tf.keras.preprocessing.image_dataset_from_directory(VAL_DIR, image_size=IMAGE_SIZE)))
steps_per_epoch = np.ceil(train_image_count / BATCH_SIZE)
validation_steps = np.ceil(val_image_count / BATCH_SIZE)

# Compile model
model_e_cbam.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
checkpoint_cb = ModelCheckpoint('models/best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')
csv_logger = CSVLogger('training_log.csv', append=True)

def build_model(name):
    if name == 'resnet50':
        model= Build_ResNet5o()
    elif name == 'efficientnetb0':
        model=EfficientNetB0_()
    else:
        raise ValueError("Unsupported model. Use 'resnet50' or 'efficientnetb0'.")
      
    return model

def main(args):
    train_ds, val_ds = get_data()
    model = build_model(args.model)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs
    )

    # Save model
    os.makedirs(args.save_dir, exist_ok=True)
    model.save(os.path.join(args.save_dir, f"{args.model}_crop_health_model.keras"))

    # Save training plot
    plot_training(history, args.save_dir)

if _name_ == '_main_':
    parser = argparse.ArgumentParser(description="Train crop health classification model")
    parser.add_argument('--model', type=str, required=True, choices=['resnet50', 'efficientnetb0'],
                        help='Model architecture: resnet50 or efficientnetb0')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save model and plots')

    args = parser.parse_args()
    main(args)
