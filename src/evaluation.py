import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, models
import math
import numpy as np
from sklearn.metrics import classification_report

sys.path.append('./src')

from EfficientNet_CBAM_Architecture import EfficientNetB0_custom  

# Load model
model_e_cbam = EfficientNetB0()
model_e_cbam.load_weights('models/efficientnet_b0_new_model.h5')

# Load dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'data/valid',
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

# Get class names
class_names = val_ds.class_names

#classification report
true_labels = []
for _, labels in val_ds.unbatch():
    true_labels.append(tf.argmax(labels).numpy())
true_labels = np.array(true_labels)

# 2. Get predicted labels
pred_labels = []
for images, _ in val_ds:
    preds = model_e_cbam.predict(images,verbose=0)
    pred_classes = np.argmax(preds, axis=1)
    pred_labels.extend(pred_classes)
pred_labels = np.array(pred_labels)

report = classification_report(true_labels, pred_labels, target_names=class_names)
print(report)
# Save to a text file
with open("classification_report.txt", "w") as f:
    f.write(report)
