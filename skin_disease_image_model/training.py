
#!/usr/bin/env python3
"""
DermNet Training Pipeline (Production Ready)
--------------------------------------------
- Combines train/test folders into single dataset
- Counts dataset images
- Builds TF ImageDataGenerator with validation split
- Trains ResNet50 classifier
- Saves & loads weights

Usage:
python dermnet_train.py     --source_dataset /data/dermnet     --combined_dataset ./combined_dataset     --epochs 20     --batch_size 32
"""
import os
import argparse
import logging
import shutil
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def combine_dataset(source_dataset_path: str, combined_dataset_path: str) -> None:
    """Combine train/test folders into a single dataset."""
    os.makedirs(combined_dataset_path, exist_ok=True)

    classes = os.listdir(os.path.join(source_dataset_path, "train"))

    for class_name in classes:
        train_class_path = os.path.join(source_dataset_path, "train", class_name)
        test_class_path = os.path.join(source_dataset_path, "test", class_name)

        combined_class_path = os.path.join(combined_dataset_path, class_name)
        os.makedirs(combined_class_path, exist_ok=True)

        for folder_path in [train_class_path, test_class_path]:
            if not os.path.exists(folder_path):
                continue

            for filename in os.listdir(folder_path):
                src = os.path.join(folder_path, filename)
                dst = os.path.join(combined_class_path, filename)

                if not os.path.exists(dst):
                    shutil.copy(src, dst)

    logger.info("Dataset combining complete.")


def count_images(dataset_folder: str) -> int:
    total_images = 0

    for root, _, files in os.walk(dataset_folder):
        total_images += len([f for f in files if f.lower().endswith((".jpg", ".png", ".jpeg"))])

    logger.info("Total images: %d", total_images)
    return total_images


def build_generators(
    dataset_directory: str,
    image_size: Tuple[int, int],
    batch_size: int,
    validation_split: float = 0.1
):
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=validation_split,
    )

    train_gen = datagen.flow_from_directory(
        dataset_directory,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        subset="training",
    )

    val_gen = datagen.flow_from_directory(
        dataset_directory,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        subset="validation",
    )

    return train_gen, val_gen


def build_model(num_classes: int, input_shape=(224, 224, 3)):
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train(
    dataset_dir: str,
    epochs: int,
    batch_size: int,
    output_weights: str
):
    image_size = (224, 224)

    train_gen, val_gen = build_generators(
        dataset_dir,
        image_size=image_size,
        batch_size=batch_size
    )

    num_classes = train_gen.num_classes
    logger.info("Num classes: %d", num_classes)

    model = build_model(num_classes)

    model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // batch_size,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=val_gen.samples // batch_size,
    )

    model.save_weights(output_weights)
    logger.info("Weights saved to %s", output_weights)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source_dataset", type=str, required=False)
    parser.add_argument("--combined_dataset", type=str, default="./combined_dataset")
    parser.add_argument("--dataset", type=str, default="./combined_dataset")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weights", type=str, default="model.weights.h5")

    args = parser.parse_args()

    if args.source_dataset:
        combine_dataset(args.source_dataset, args.combined_dataset)

    count_images(args.dataset)

    train(
        dataset_dir=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_weights=args.weights,
    )


if __name__ == "__main__":
    main()
