import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_cv

# Set paths and configurations
DATASET_DIR = os.path.join(os.getcwd(), "data")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# Set up directories for train, validation, and test images and masks
train_images_dir = os.path.join(TRAIN_DIR, "images")
train_masks_dir = os.path.join(TRAIN_DIR, "masks")
val_images_dir = os.path.join(VAL_DIR, "images")
val_masks_dir = os.path.join(VAL_DIR, "masks")

# Calculate the number of images in each split for setting batch size dynamically
train_batch_size = len(os.listdir(train_images_dir))
val_batch_size = len(os.listdir(val_images_dir))

# Hyperparameters and configuration
EPOCHS = 50  # Adjust as needed
IMG_HEIGHT = 512
IMG_WIDTH = 512
NUM_CLASSES = 2  # Adjust this according to your dataset

# Function to load images and masks
def load_images_and_masks(images_dir, masks_dir, img_size=(IMG_HEIGHT, IMG_WIDTH)):
    image_paths = sorted([os.path.join(images_dir, fname) for fname in os.listdir(images_dir) if fname.endswith(".jpg") or fname.endswith(".png")])
    mask_paths = sorted([os.path.join(masks_dir, fname) for fname in os.listdir(masks_dir) if fname.endswith(".jpg") or fname.endswith(".png")])

    images = np.zeros((len(image_paths), img_size[0], img_size[1], 3), dtype=np.uint8)
    masks = np.zeros((len(mask_paths), img_size[0], img_size[1], 1), dtype=np.uint8)

    for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
        img = tf.keras.preprocessing.image.img_to_array(img)
        mask = tf.keras.preprocessing.image.load_img(mask_path, color_mode="grayscale", target_size=img_size)
        mask = tf.keras.preprocessing.image.img_to_array(mask)

        images[i] = img
        masks[i] = mask

    return images, masks

print('loading datasets')
# Load training and validation datasets
train_images, train_masks = load_images_and_masks(train_images_dir, train_masks_dir)
val_images, val_masks = load_images_and_masks(val_images_dir, val_masks_dir)

print('normalize datasets')
# Normalize images and masks
train_images = train_images / 255.0
val_images = val_images / 255.0
train_masks = train_masks / 255.0
val_masks = val_masks / 255.0

# Ensure masks are in correct format
train_masks = np.round(train_masks).astype(np.uint8)
val_masks = np.round(val_masks).astype(np.uint8)

print('prep datasets')
# Prepare data generators
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
train_dataset = train_dataset.batch(train_batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_masks))
val_dataset = val_dataset.batch(val_batch_size)

# Use a compatible backbone provided by KerasCV
backbone = keras_cv.models.ResNet50Backbone(
    include_rescaling=False,  # Set to True if input data needs to be rescaled
)

# Define DeepLabV3+ model with KerasCV using the compatible backbone
base_model = keras_cv.models.segmentation.DeepLabV3Plus(
    backbone=backbone,
    num_classes=NUM_CLASSES,
)

print('compiling')
# Compile model
base_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

print('training will start')
# Train the model
base_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
)
print('training done')

# Save the model after training
MODEL_SAVE_DIR = os.path.join(os.getcwd(), "dlv3plus", "models")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
base_model.save(os.path.join(MODEL_SAVE_DIR, "deeplabv3plus_model.h5"))

print("Training completed and model saved.")
