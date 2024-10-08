import os
import sys
import numpy as np
import tensorflow as tf
import keras_cv
from datetime import datetime

DATASET_DIR = os.path.join(os.getcwd(), "data")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

LOG_SAVE_DIR = os.path.join(os.getcwd(), "dlv3plus", "logs")
MODEL_SAVE_DIR = os.path.join(os.getcwd(), "dlv3plus", "models")

BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'best_model.weights.h5')
BEST_VAL_LOSS_FILE = os.path.join(MODEL_SAVE_DIR, "best_val_loss.txt")
EPOCH_FILE_PATH = os.path.join(MODEL_SAVE_DIR, 'last_epoch.txt')
initial_epoch = 0

if os.path.exists(BEST_VAL_LOSS_FILE):
    with open(BEST_VAL_LOSS_FILE, 'r') as f:
        best_val_loss = float(f.read().strip())
else:
    best_val_loss = np.inf

def read_last_epoch(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return int(f.read().strip())
    return 0

def write_last_epoch(file_path, epoch):
    with open(file_path, 'w') as f:
        f.write(str(epoch))

train_images_dir = os.path.join(TRAIN_DIR, "images")
train_masks_dir = os.path.join(TRAIN_DIR, "masks")
val_images_dir = os.path.join(VAL_DIR, "images")
val_masks_dir = os.path.join(VAL_DIR, "masks")

train_batch_size = 1
val_batch_size = 1

EPOCHS = 100 
IMAGE_MIN_DIM = 800
IMAGE_MAX_DIM = 1024
NUM_CLASSES = 2
LEARNING_RATE = 0.001

def load_image(image_path, mask=False):
    img = tf.keras.preprocessing.image.load_img(image_path.numpy().decode("utf-8"), color_mode="grayscale" if mask else "rgb")
    img = tf.keras.preprocessing.image.img_to_array(img)
    return img

def parse_function(image_path, mask_path):
    image = tf.py_function(load_image, [image_path], tf.float32)
    mask = tf.py_function(load_image, [mask_path, True], tf.float32)

    image.set_shape([None, None, 3])
    mask.set_shape([None, None, 1])
    
    image = tf.image.resize(image, [IMAGE_MAX_DIM, IMAGE_MAX_DIM], method='bilinear')
    mask = tf.image.resize(mask, [IMAGE_MAX_DIM, IMAGE_MAX_DIM], method='nearest')
    
    image = image / 255.0
    mask = mask / 255.0

    return image, mask

def create_dataset(image_paths, mask_paths, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

train_image_paths = sorted([os.path.join(train_images_dir, fname) for fname in os.listdir(train_images_dir) if fname.endswith(".jpg") or fname.endswith(".png")])
train_mask_paths = sorted([os.path.join(train_masks_dir, fname) for fname in os.listdir(train_masks_dir) if fname.endswith(".jpg") or fname.endswith(".png")])

val_image_paths = sorted([os.path.join(val_images_dir, fname) for fname in os.listdir(val_images_dir) if fname.endswith(".jpg") or fname.endswith(".png")])
val_mask_paths = sorted([os.path.join(val_masks_dir, fname) for fname in os.listdir(val_masks_dir) if fname.endswith(".jpg") or fname.endswith(".png")])

train_dataset = create_dataset(train_image_paths, train_mask_paths, train_batch_size)
val_dataset = create_dataset(val_image_paths, val_mask_paths, val_batch_size)

backbone = keras_cv.models.ResNet50Backbone(
    include_rescaling=False,
)

base_model = keras_cv.models.segmentation.DeepLabV3Plus(
    backbone=backbone,
    num_classes=NUM_CLASSES,
)

base_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[
        "accuracy",
    ]
)

class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_file, model_dir, weights_dir):
        super(MetricsCallback, self).__init__()
        self.validation_data = val_dataset
        self.log_file = log_file
        self.start_time = None
        self.best_mean_iou = 0.0
        self.best_checkpoint_path = os.path.join(weights_dir, "best_model.weights.h5")
        self.model_dir = model_dir
        self.weights_dir = weights_dir
        self.epoch_file_path = EPOCH_FILE_PATH
        self.best_val_loss = best_val_loss

        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write("start_time,epoch,end_time,epoch_duration,loss,val_loss,mean_iou,mean_precision,mean_recall,mean_f1_score\n")

    def on_train_begin(self, logs=None):
        tf.keras.backend.clear_session(free_memory=True)
        self.start_time = datetime.now()
        formatted_start_time = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Training started at: {formatted_start_time}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f'\nEpoch {epoch + 1} Metrics:')
        end_time = datetime.now()
        epoch_duration = (end_time - self.start_time).total_seconds()
        formatted_end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")

        loss = logs.get('loss', 'N/A')
        val_loss = logs.get('val_loss', 'N/A')

        if val_loss is not None and val_loss < self.best_val_loss:
            print(f"Validation loss improved from {self.best_val_loss} to {val_loss}. Saving best model.")
            self.best_val_loss = val_loss

            self.model.save_weights(os.path.join(self.weights_dir, "best_model.weights.h5"))

            with open(BEST_VAL_LOSS_FILE, 'w') as f:
                f.write(str(self.best_val_loss))

        write_last_epoch(self.epoch_file_path, epoch + 1)

        val_iou, precisions, recalls, f1_scores = [], [], [], []

        for val_images, val_masks in self.validation_data:
            preds = self.model.predict(val_images)
            preds = np.argmax(preds, axis=-1)
            preds = np.expand_dims(preds, axis=-1)

            for i in range(len(val_images)):
                pred_mask = preds[i]
                true_mask = val_masks[i]

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

        mean_iou_value = np.mean(val_iou)
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)
        mean_f1_score = np.mean(f1_scores)

        logs['mean_iou'] = mean_iou_value
        logs['mean_precision'] = mean_precision
        logs['mean_recall'] = mean_recall
        logs['mean_f1_score'] = mean_f1_score

        with open(self.log_file, 'a') as f:
            f.write(f"{self.start_time.strftime('%Y-%m-%d %H:%M:%S')},{epoch + 1},{formatted_end_time},{epoch_duration:.2f},{loss:.4f},{val_loss:.4f},{mean_iou_value:.4f},{mean_precision:.4f},{mean_recall:.4f},{mean_f1_score:.4f}\n")

        self.start_time = datetime.now()

        for metric_name, metric_value in logs.items():
            print(f'{metric_name}: {metric_value:.4f}')
        
        print("Epoch completed, terminating the process to restart.")
        sys.exit()

os.makedirs(LOG_SAVE_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

log_file_path = os.path.join(LOG_SAVE_DIR, "training_log.csv")

metrics_callback = MetricsCallback(log_file=log_file_path, model_dir=MODEL_SAVE_DIR, weights_dir=MODEL_SAVE_DIR)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(MODEL_SAVE_DIR, 'ckpt_epoch_{epoch:04d}.weights.h5'),
    save_weights_only=True,
    save_best_only=False,
    save_freq='epoch',
    verbose=1
)

if os.path.exists(BEST_MODEL_PATH):
    print(f"Loading best model from {BEST_MODEL_PATH}")
    base_model.load_weights(BEST_MODEL_PATH)
    
    initial_epoch = read_last_epoch(EPOCH_FILE_PATH)
else:
    print("No best model found, starting from epoch 0")
    initial_epoch = 0


print('Training will start')
base_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    initial_epoch=initial_epoch,
    callbacks=[checkpoint_callback, metrics_callback]
)
print('Training done')
