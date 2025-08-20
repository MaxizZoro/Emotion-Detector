import tensorflow as tf
import os
import matplotlib.pyplot as plt

# Set dataset paths
data_dir = "../data/FER-2013"
batch_size = 32
img_size = (48, 48)

# Load training and test datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(data_dir, "train"),
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(data_dir, "test"),
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

# Optional: class names
class_names = train_ds.class_names
print("Emotion classes:", class_names)

# Normalize pixel values to [0, 1]
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# Prefetch and cache for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Visualize 9 sample images from a training batch
for images, labels in train_ds.take(1):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("float32"), cmap='gray')
        plt.title(class_names[labels[i]])
        plt.axis("off")
    plt.show()