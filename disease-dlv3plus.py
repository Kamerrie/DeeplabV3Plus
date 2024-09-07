import os
import sys
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

BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'best_model.weights.h5')
BEST_IOU_FILE = os.path.join(MODEL_SAVE_DIR, "best_mean_iou.txt")
WAIT_COUNTER_FILE = os.path.join(MODEL_SAVE_DIR, 'wait_counter.txt')
EPOCH_FILE_PATH = os.path.join(MODEL_SAVE_DIR, 'last_epoch.txt')

# Initialize best IoU and wait counter
if os.path.exists(BEST_IOU_FILE):
    with open(BEST_IOU_FILE, 'r') as f:
        best_mean_iou = float(f.read().strip())
else:
    best_mean_iou = 0.0

if os.path.exists(WAIT_COUNTER_FILE):
    with open(WAIT_COUNTER_FILE, 'r') as f:
        wait_counter = int(f.read().strip())
else:
    wait_counter = 0

# Helper functions to read/write epoch and wait counter
def read_last_epoch(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return int(f.read().strip())
    return 0

def write_last_epoch(file_path, epoch):
    with open(file_path, 'w') as f:
        f.write(str(epoch))

def read_wait_counter(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return int(f.read().strip())
    return 0

def write_wait_counter(file_path, wait):
    with open(file_path, 'w') as f:
        f.write(str(wait))

# Set up directories for train, validation, and test images and masks
train_images_dir = os.path.join(TRAIN_DIR, "images")
train_masks_dir = os.path.join(TRAIN_DIR, "masks")
val_images_dir = os.path.join(VAL_DIR, "images")
val_masks_dir = os.path.join(VAL_DIR, "masks")

# Calculate the number of images in each split for setting batch size dynamically
train_batch_size = 1
val_batch_size = 1

# Hyperparameters and configuration
EPOCHS = 100  # Adjust as needed
IMAGE_MIN_DIM = 800
IMAGE_MAX_DIM = 1024
NUM_CLASSES = 2  # Adjust this according to your dataset
LEARNING_RATE = 0.001

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
# Compile model with additional metrics
base_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[
        "accuracy",
    ]
)

# Define the custom callback for saving based on mean IoU and managing early stopping
class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_file, model_dir, weights_dir, patience=10):
        super(MetricsCallback, self).__init__()
        self.validation_data = val_dataset
        self.log_file = log_file
        self.start_time = None
        self.best_mean_iou = best_mean_iou  # Initialize with the best IoU from file
        self.best_checkpoint_path = os.path.join(weights_dir, "best_model.weights.h5")  # Path to save the best model
        self.model_dir = model_dir
        self.weights_dir = weights_dir
        self.epoch_file_path = EPOCH_FILE_PATH
        self.patience = patience  # Early stopping patience
        self.wait = wait_counter  # Initialize the wait counter from file

        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write("start_time,epoch,end_time,epoch_duration,loss,val_loss,mean_iou,mean_precision,mean_recall,mean_f1_score\n")

    def on_train_begin(self, logs=None):
        self.start_time = datetime.now()
        formatted_start_time = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Training started at: {formatted_start_time}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f'\nEpoch {epoch + 1} Metrics:')
        end_time = datetime.now()
        epoch_duration = (end_time - self.start_time).total_seconds()
        formatted_end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")

        # Save the last completed epoch
        write_last_epoch(self.epoch_file_path, epoch + 1)

        # Evaluate model on validation data and compute custom metrics
        val_iou, precisions, recalls, f1_scores = [], [], [], []

        for val_images, val_masks in self.validation_data:
            preds = self.model.predict(val_images)
            preds = np.argmax(preds, axis=-1)  # Convert predictions to class indices
            preds = np.expand_dims(preds, axis=-1)  # Ensure prediction shape matches true mask

            for i in range(len(val_images)):
                pred_mask = preds[i]
                true_mask = val_masks[i]

                # Ensure compatibility for logical operations
                pred_mask = np.squeeze(pred_mask)
                true_mask = np.squeeze(true_mask)

                intersection = np.logical_and(pred_mask, true_mask)
                union = np.logical_or(pred_mask, true_mask)
                iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
                val_iou.append(iou)

                precision = np.sum(intersection) / np.sum(pred_mask) if np.sum(pred_mask) > 0 else 0
                recall = np.sum(intersection) / np.sum(true_mask) if np.sum(true_mask) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)

        # Calculate mean values for metrics
        mean_iou_value = np.mean(val_iou)
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)
        mean_f1_score = np.mean(f1_scores)

        # Update logs dictionary
        logs['mean_iou'] = mean_iou_value
        logs['mean_precision'] = mean_precision
        logs['mean_recall'] = mean_recall
        logs['mean_f1_score'] = mean_f1_score

        # Check if the mean IoU improved
        if mean_iou_value > self.best_mean_iou:
            print(f"Mean IoU improved from {self.best_mean_iou} to {mean_iou_value}. Saving best model.")
            self.best_mean_iou = mean_iou_value
            self.wait = 0  # Reset wait counter if IoU improves

            # Save the best model
            self.model.save_weights(self.best_checkpoint_path)

            # Persist the best IoU to a file
            with open(BEST_IOU_FILE, 'w') as f:
                f.write(str(self.best_mean_iou))

        else:
            self.wait += 1  # Increment the wait counter
            write_wait_counter(WAIT_COUNTER_FILE, self.wait)  # Persist the wait counter

            if self.wait >= self.patience:
                print(f"Stopping training after {self.wait} epochs without improvement in IoU.")
                self.model.stop_training = True

        # Log metrics to file
        with open(self.log_file, 'a') as f:
            f.write(f"{self.start_time.strftime('%Y-%m-%d %H:%M:%S')},{epoch + 1},{formatted_end_time},{epoch_duration:.2f},{logs.get('loss', 'N/A'):.4f},{logs.get('val_loss', 'N/A'):.4f},{mean_iou_value:.4f},{mean_precision:.4f},{mean_recall:.4f},{mean_f1_score:.4f}\n")

        self.start_time = datetime.now()

        # Print all metrics
        for metric_name, metric_value in logs.items():
            print(f'{metric_name}: {metric_value:.4f}')


# Set up the log and model directories
os.makedirs(LOG_SAVE_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

log_file_path = os.path.join(LOG_SAVE_DIR, "training_log.csv")

# Instantiate the custom callback
metrics_callback = MetricsCallback(log_file=log_file_path, model_dir=MODEL_SAVE_DIR, weights_dir=MODEL_SAVE_DIR)

# Define a checkpoint callback to save TensorFlow checkpoints
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(MODEL_SAVE_DIR, 'ckpt_epoch_{epoch:04d}.weights.h5'),
    save_weights_only=True,
    save_best_only=False,
    save_freq='epoch',
    verbose=1
)

# Check for the best model file and the last epoch
if os.path.exists(BEST_MODEL_PATH):
    print(f"Loading best model from {BEST_MODEL_PATH}")
    base_model.load_weights(BEST_MODEL_PATH)
    initial_epoch = read_last_epoch(EPOCH_FILE_PATH)
else:
    print("No best model found, starting from epoch 0")
    initial_epoch = 0

print('Training will start')
# Start training the model
base_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    initial_epoch=initial_epoch,
    callbacks=[checkpoint_callback, metrics_callback]
)
print('Training done')
