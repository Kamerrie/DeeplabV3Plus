import os
import numpy as np
import tensorflow as tf
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
def load_image(image_path, mask=False):
    img = tf.keras.preprocessing.image.load_img(image_path.numpy().decode("utf-8"), color_mode="grayscale" if mask else "rgb")
    img = tf.keras.preprocessing.image.img_to_array(img)
    return img

def parse_function(image_path, mask_path):
    image = tf.py_function(load_image, [image_path], tf.float32)
    mask = tf.py_function(load_image, [mask_path, True], tf.float32)

    # Set the shape of the images manually
    image.set_shape([None, None, 3])
    mask.set_shape([None, None, 1])
    
    # Resize image and mask to a consistent size (IMAGE_MAX_DIM x IMAGE_MAX_DIM)
    image = tf.image.resize(image, [IMAGE_MAX_DIM, IMAGE_MAX_DIM], method='bilinear')
    mask = tf.image.resize(mask, [IMAGE_MAX_DIM, IMAGE_MAX_DIM], method='nearest')
    
    # Normalize images and masks
    image = image / 255.0
    mask = mask / 255.0

    return image, mask

def create_dataset(image_paths, mask_paths, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# Get image and mask paths
train_image_paths = sorted([os.path.join(train_images_dir, fname) for fname in os.listdir(train_images_dir) if fname.endswith(".jpg") or fname.endswith(".png")])
train_mask_paths = sorted([os.path.join(train_masks_dir, fname) for fname in os.listdir(train_masks_dir) if fname.endswith(".jpg") or fname.endswith(".png")])

val_image_paths = sorted([os.path.join(val_images_dir, fname) for fname in os.listdir(val_images_dir) if fname.endswith(".jpg") or fname.endswith(".png")])
val_mask_paths = sorted([os.path.join(val_masks_dir, fname) for fname in os.listdir(val_masks_dir) if fname.endswith(".jpg") or fname.endswith(".png")])

print('loading datasets')
# Create TensorFlow datasets for training and validation
train_dataset = create_dataset(train_image_paths, train_mask_paths, train_batch_size)
val_dataset = create_dataset(val_image_paths, val_mask_paths, val_batch_size)

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
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# Custom callback to log metrics to a file and print them
class MetricsCallback(tf.keras.callbacks.Callback):
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
