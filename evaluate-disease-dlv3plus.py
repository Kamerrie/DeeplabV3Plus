import os
import numpy as np
import tensorflow as tf
import keras_cv
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, jaccard_score
import seaborn as sns

DATASET_DIR = os.path.join(os.getcwd(), "data")
TEST_DIR = os.path.join(DATASET_DIR, "test")

MODEL_SAVE_DIR = os.path.join(os.getcwd(), "dlv3plus", "models")
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'best_model.weights.h5')
VISUALIZATION_DIR = os.path.join(os.getcwd(), "dlv3plus", "visualizations")

os.makedirs(VISUALIZATION_DIR, exist_ok=True)

test_images_dir = os.path.join(TEST_DIR, "images")
test_masks_dir = os.path.join(TEST_DIR, "masks")

test_batch_size = 1

def load_image(image_path, mask=False):
    img = tf.keras.preprocessing.image.load_img(image_path.numpy().decode("utf-8"), color_mode="grayscale" if mask else "rgb")
    img = tf.keras.preprocessing.image.img_to_array(img)
    return img

def parse_function(image_path, mask_path):
    image = tf.py_function(load_image, [image_path], tf.float32)
    mask = tf.py_function(load_image, [mask_path, True], tf.float32)

    image.set_shape([None, None, 3])
    mask.set_shape([None, None, 1])
    
    IMAGE_MAX_DIM = 1024
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

test_image_paths = sorted([os.path.join(test_images_dir, fname) for fname in os.listdir(test_images_dir) if fname.endswith(".jpg") or fname.endswith(".png")])
test_mask_paths = sorted([os.path.join(test_masks_dir, fname) for fname in os.listdir(test_masks_dir) if fname.endswith(".jpg") or fname.endswith(".png")])

test_dataset = create_dataset(test_image_paths, test_mask_paths, test_batch_size)

backbone = keras_cv.models.ResNet50Backbone(
    include_rescaling=False,
)

model = keras_cv.models.segmentation.DeepLabV3Plus(
    backbone=backbone,
    num_classes=2,
)

if os.path.exists(BEST_MODEL_PATH):
    print(f"Loading best model from {BEST_MODEL_PATH}")
    model.load_weights(BEST_MODEL_PATH)
else:
    raise FileNotFoundError(f"No best model found at {BEST_MODEL_PATH}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)

print("Evaluating model...")
results = model.evaluate(test_dataset, return_dict=True)

print("\nEvaluation Results:")
for metric_name, metric_value in results.items():
    print(f"{metric_name}: {metric_value:.4f}")

def visualize_and_save_results(test_images, test_masks, preds, idx):
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    ax[0].imshow(test_images[0])
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(np.squeeze(test_masks[0]), cmap="gray")
    ax[1].set_title("True Mask")
    ax[1].axis("off")

    ax[2].imshow(np.squeeze(preds[0]), cmap="gray")
    ax[2].set_title("Predicted Mask")
    ax[2].axis("off")

    ax[3].imshow(test_images[0])
    ax[3].imshow(np.squeeze(preds[0]), cmap="gray", alpha=0.5, interpolation='none')
    ax[3].set_title("Overlay of Predicted Mask")
    ax[3].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    save_path = os.path.join(VISUALIZATION_DIR, f"result_{idx:04d}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved visualization to {save_path}")

print("Generating predictions and saving visualizations...")
for idx, (test_images, test_masks) in enumerate(test_dataset):
    preds = model.predict(test_images)
    preds = np.argmax(preds, axis=-1)
    preds = np.expand_dims(preds, axis=-1)

    visualize_and_save_results(test_images.numpy(), test_masks.numpy(), preds, idx)

print("All visualizations saved.")

all_preds = []
all_trues = []

for test_images, test_masks in test_dataset:
    preds = model.predict(test_images)
    preds = np.argmax(preds, axis=-1)
    preds = np.expand_dims(preds, axis=-1)

    all_preds.append(preds)
    all_trues.append(test_masks.numpy())

all_preds = np.concatenate(all_preds, axis=0).flatten()
all_trues = np.concatenate(all_trues, axis=0).flatten()

print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(all_trues, all_preds)
print(conf_matrix)

conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

TN, FP, FN, TP = conf_matrix.ravel()

print(f"True Positives (TP): {TP}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")
print(f"True Negatives (TN): {TN}")

iou = jaccard_score(all_trues, all_preds, average='binary')
print(f"\nIoU (Jaccard Index): {iou:.4f}")

labels = np.array([
    [f"TN\n{conf_matrix[0, 0]} ({conf_matrix_percentage[0, 0]:.1f}%)", f"FP\n{conf_matrix[0, 1]} ({conf_matrix_percentage[0, 1]:.1f}%)"],
    [f"FN\n{conf_matrix[1, 0]} ({conf_matrix_percentage[1, 0]:.1f}%)", f"TP\n{conf_matrix[1, 1]} ({conf_matrix_percentage[1, 1]:.1f}%)"]
])

plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix, 
    annot=labels, 
    fmt='', 
    cmap='Blues', 
    xticklabels=["Background", "Disease"], 
    yticklabels=["Background", "Disease"]
)
plt.xlabel('Predicted')
plt.ylabel('Ground Truth')
plt.title('Confusion Matrix')
conf_matrix_path = os.path.join(VISUALIZATION_DIR, 'confusion_matrix.png')
plt.savefig(conf_matrix_path)
plt.close()
print(f"Saved confusion matrix with counts and percentages to {conf_matrix_path}")

results_file_path = os.path.join(MODEL_SAVE_DIR, 'evaluation_results.txt')
with open(results_file_path, 'w') as f:
    for metric_name, metric_value in results.items():
        f.write(f"{metric_name}: {metric_value:.4f}\n")

with open(results_file_path, 'a') as f:
    f.write("\nConfusion Matrix:\n")
    f.write(np.array2string(conf_matrix))
    f.write(f"\n\nTrue Positives (TP): {TP}\n")
    f.write(f"False Positives (FP): {FP}\n")
    f.write(f"False Negatives (FN): {FN}\n")
    f.write(f"True Negatives (TN): {TN}\n")
    f.write(f"IoU (Jaccard Index): {iou:.4f}\n")
    f.write("\n\nClassification Report:\n")
    f.write(classification_report(all_trues, all_preds, target_names=["Background", "Disease"]))

print("All evaluation metrics saved.")
