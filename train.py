import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# Dataset paths
metadata_path = "dataset/HAM10000_metadata.csv"
image_folders = [
    "dataset/HAM10000_images_part_1",
    "dataset/HAM10000_images_part_2"
]

# Read metadata
df = pd.read_csv(metadata_path)

print("Total Images:", len(df))
print("Classes:")
print(df["dx"].value_counts())

# Find image path
def get_image_path(image_id):
    for folder in image_folders:
        path = os.path.join(folder, image_id + ".jpg")
        if os.path.exists(path):
            return path
    return None

df["image_path"] = df["image_id"].apply(get_image_path)

# Remove missing images
df = df[df["image_path"].notnull()]

print("Images Found:", len(df))

# Encode labels
class_names = sorted(df["dx"].unique())
label_map = {name: idx for idx, name in enumerate(class_names)}

df["label"] = df["dx"].map(label_map)

print("Label Map:")
print(label_map)

# Split dataset
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print("Training Images:", len(train_df))
print("Validation Images:", len(val_df))

import tensorflow as tf
import numpy as np

IMG_SIZE = 224
BATCH_SIZE = 32

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    return image, label

train_ds = tf.data.Dataset.from_tensor_slices(
    (train_df["image_path"].values, train_df["label"].values)
)

val_ds = tf.data.Dataset.from_tensor_slices(
    (val_df["image_path"].values, val_df["label"].values)
)

train_ds = train_ds.map(load_image).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(load_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224,224,3)),

    tf.keras.layers.Conv2D(32,3,activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64,3,activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128,3,activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128,activation="relu"),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(7,activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

os.makedirs("model", exist_ok=True)
model.save("model/skin_disease_model.h5")

print("✅ Model Saved Successfully!")