import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_cv
from datetime import datetime

# Set paths and configurations
DATASET_DIR = os.path.join(os.getcwd(), "data")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

LOG_SAVE_DIR = os.path.join(os.getcwd(), "dlv3plus", "logs")
MODEL_SAVE_DIR = os.path.join(os.getcwd(), "dlv3plus", "models")

# Set up directories for train, validation, and test images and masks
train_images_dir = os.path.join(TRAIN_DIR, "images")
train_masks_dir = os.path.join(TRAIN_DIR, "masks")
val_images_dir = os.path.join(VAL_DIR, "images")
val_masks_dir = os.path.join(VAL_DIR, "masks")

# Calculate the number of images in each split for setting batch size dynamically
train_batch_size = len(os.listdir(train_images_dir))
val_batch_size = len(os.listdir(val_images_dir))

# Hyperparameters and configuration
EPOCHS = 100  # Adjust as needed
IMAGE_MIN_DIM = 800
IMAGE_MAX_DIM = 1024
NUM_CLASSES = 2  # Adjust this according to your dataset

# Function to load and resize images and masks according to configuration
def load_images_and_masks(images_dir, masks_dir, min_dim=IMAGE_MIN_DIM, max_dim=IMAGE_MAX_DIM):
    image_paths = sorted([os.path.join(images_dir, fname) for fname in os.listdir(images_dir) if fname.endswith(".jpg") or fname.endswith(".png")])
    mask_paths = sorted([os.path.join(masks_dir, fname) for fname in os.listdir(masks_dir) if fname.endswith(".jpg") or fname.endswith(".png")])

    # Initialize lists to store resized images and masks
    images = []
    masks = []

    for img_path, mask_path in zip(image_paths, mask_paths):
        img = tf.keras.preprocessing.image.load_img(img_path)
        img = tf.keras.preprocessing.image.img_to_array(img)

        mask = tf.keras.preprocessing.image.load_img(mask_path, color_mode="grayscale")
        mask = tf.keras.preprocessing.image.img_to_array(mask)

        # Resize image and mask according to the IMAGE_MIN_DIM and IMAGE_MAX_DIM
        img = resize_and_pad_image(img, min_dim, max_dim)
        mask = resize_and_pad_image(mask, min_dim, max_dim, is_mask=True)

        # Append resized and padded images and masks to the list
        images.append(img)
        masks.append(mask)

    # Convert lists to numpy arrays
    images = np.array(images, dtype=np.uint8)
    masks = np.array(masks, dtype=np.uint8)

    return images, masks

def resize_and_pad_image(image, min_dim, max_dim, is_mask=False):
    """
    Resize an image keeping the aspect ratio unchanged and pad it to have a square shape.
    """
    image_shape = image.shape[:2]
    scale = min_dim / min(image_shape)
    if max(image_shape) * scale > max_dim:
        scale = max_dim / max(image_shape)
    
    new_size = (int(round(image_shape[1] * scale)), int(round(image_shape[0] * scale)))
    image = tf.image.resize(image, new_size, method='nearest' if is_mask else 'bilinear')

    # Padding to square
    delta_w = max_dim - new_size[0]
    delta_h = max_dim - new_size[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # Correct padding argument format
    paddings = [[top, bottom], [left, right], [0, 0]]
    image = tf.pad(image, paddings, "CONSTANT", constant_values=0)

    return image

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

# Custom callback to log metrics to a file and print them
class MetricsCallback(keras.callbacks.Callback):
    def __init__(self, log_file):
        super(MetricsCallback, self).__init__()
        self.log_file = log_file
        self.start_time = None
        
        # Initialize the log file with headers if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write("start_time,epoch,end_time,epoch_duration,loss,val_loss,mean_iou,mean_precision,mean_recall,mean_f1_score\n")

    def on_epoch_begin(self, epoch, logs=None):
        # Record the start time of the epoch
        self.start_time = datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        end_time = datetime.now()
        epoch_duration = (end_time - self.start_time).total_seconds()
        start_time_formatted = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
        end_time_formatted = end_time.strftime("%Y-%m-%d %H:%M:%S")

        # Calculate additional metrics if needed (e.g., IoU, precision, recall, F1 score)
        mean_iou = logs.get('mean_iou', 'N/A')
        mean_precision = logs.get('mean_precision', 'N/A')
        mean_recall = logs.get('mean_recall', 'N/A')
        mean_f1_score = logs.get('mean_f1_score', 'N/A')
        loss = logs.get('loss', 'N/A')
        val_loss = logs.get('val_loss', 'N/A')

        # Print logs to screen
        print(f"Epoch {epoch + 1} - Logs: {logs}")

        # Log the metrics to the file
        with open(self.log_file, 'a') as f:
            f.write(f"{start_time_formatted},{epoch + 1},{end_time_formatted},{epoch_duration:.2f},{loss},{val_loss},{mean_iou},{mean_precision},{mean_recall},{mean_f1_score}\n")

        print(f"Logged metrics for epoch {epoch + 1}")

os.makedirs(LOG_SAVE_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_SAVE_DIR, "training_log.csv")

# Callback instance
metrics_callback = MetricsCallback(log_file=log_file_path)

print('training will start')
# Train the model with the custom callback
base_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[metrics_callback]
)
print('training done')

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
base_model.save(os.path.join(MODEL_SAVE_DIR, "deeplabv3plus_model.h5"))

print("Training completed and model saved.")
