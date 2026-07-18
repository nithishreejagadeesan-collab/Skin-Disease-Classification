import tensorflow as tf

train_dir = "dataset/SkinDisease/SkinDisease/train"
test_dir = "dataset/SkinDisease/SkinDisease/test"

train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(224,224),
    batch_size=32
)

test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(224,224),
    batch_size=32
)

print("Classes:")
print(train_data.class_names)

print("Dataset loaded successfully!")
