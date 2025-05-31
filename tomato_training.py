"""
Tomato Classification Training Script
Huấn luyện model phân loại cà chua chín/xanh sử dụng MobileNet-v3
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime
import logging
from typing import Tuple, Dict, Optional, Any
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TomatoTrainer:
    def __init__(self, img_size: Tuple[int, int] = (224, 224), num_classes: int = 2):
        """
        Initialize the TomatoTrainer.

        Args:
            img_size: Tuple of (height, width) for input images
            num_classes: Number of output classes
        """
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.class_names = ['green', 'ripe']  # 0: xanh, 1: chín

    def create_model(self) -> tf.keras.Model:
        """
        Create model using MobileNet-v3 Large with transfer learning.

        Returns:
            tf.keras.Model: The compiled model
        """
        logger.info("Creating MobileNet-v3 model...")

        try:
            # Load pre-trained MobileNet-v3
            base_model = MobileNetV3Large(
                input_shape=(*self.img_size, 3),
                include_top=False,
                weights='imagenet'
            )

            # Freeze base model layers
            base_model.trainable = False

            # Add custom classification layers
            self.model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.2),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.1),
                layers.Dense(self.num_classes, activation='softmax')
            ])

            # Compile model
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            # Print model summary
            logger.info("\nModel Architecture:")
            self.model.summary()

            return self.model

        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            raise

    def prepare_data_generators(
        self, 
        train_dir: str, 
        val_dir: str, 
        batch_size: int = 32
    ) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
        """
        Prepare data generators for training and validation.

        Args:
            train_dir: Path to training data directory
            val_dir: Path to validation data directory
            batch_size: Batch size for training

        Returns:
            Tuple of (train_generator, val_generator)
        """
        logger.info(f"Preparing data from {train_dir} and {val_dir}...")

        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            raise ValueError("Training or validation directory does not exist")

        try:
            # Data augmentation for training
            train_datagen = ImageDataGenerator(
                rescale=1. / 255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                shear_range=0.1,
                fill_mode='nearest'
            )

            # Only rescale for validation
            val_datagen = ImageDataGenerator(rescale=1. / 255)

            # Create generators
            train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=self.img_size,
                batch_size=batch_size,
                class_mode='categorical',
                classes=self.class_names,
                shuffle=True
            )

            val_generator = val_datagen.flow_from_directory(
                val_dir,
                target_size=self.img_size,
                batch_size=batch_size,
                class_mode='categorical',
                classes=self.class_names,
                shuffle=False
            )

            logger.info(f"Training images: {train_generator.samples}")
            logger.info(f"Validation images: {val_generator.samples}")
            logger.info(f"Classes: {train_generator.class_indices}")

            return train_generator, val_generator

        except Exception as e:
            logger.error(f"Error preparing data generators: {str(e)}")
            raise

    def setup_callbacks(self, model_save_path: str) -> list:
        """
        Setup callbacks for training.

        Args:
            model_save_path: Path to save the model

        Returns:
            List of callbacks
        """
        try:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=7,
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=3,
                    min_lr=0.0001,
                    verbose=1
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    model_save_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                ),
                tf.keras.callbacks.CSVLogger(
                    f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
                )
            ]
            return callbacks
        except Exception as e:
            logger.error(f"Error setting up callbacks: {str(e)}")
            raise

    def train_model(
        self,
        train_generator: ImageDataGenerator,
        val_generator: ImageDataGenerator,
        epochs: int = 25,
        model_save_path: str = 'best_tomato_model.h5'
    ) -> tf.keras.callbacks.History:
        """
        Train the model.

        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs: Number of training epochs
            model_save_path: Path to save the best model

        Returns:
            Training history
        """
        if self.model is None:
            self.create_model()

        logger.info(f"\n=== STARTING TRAINING ===")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Model will be saved at: {model_save_path}")

        try:
            # Setup callbacks
            callbacks = self.setup_callbacks(model_save_path)

            # Training
            history = self.model.fit(
                train_generator,
                epochs=epochs,
                validation_data=val_generator,
                callbacks=callbacks,
                verbose=1
            )

            logger.info("\n=== TRAINING COMPLETED ===")
            return history

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def fine_tune_model(
        self,
        train_generator: ImageDataGenerator,
        val_generator: ImageDataGenerator,
        epochs: int = 10,
        model_save_path: str = 'fine_tuned_tomato_model.h5'
    ) -> Optional[tf.keras.callbacks.History]:
        """
        Fine-tune the model by unfreezing some top layers.

        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs: Number of fine-tuning epochs
            model_save_path: Path to save the fine-tuned model

        Returns:
            Fine-tuning history if successful, None otherwise
        """
        if self.model is None:
            logger.error("Model not created. Please train the model first.")
            return None

        logger.info("\n=== STARTING FINE-TUNING ===")

        try:
            # Unfreeze top layers of base model
            base_model = self.model.layers[0]
            base_model.trainable = True

            # Freeze early layers, only fine-tune later layers
            fine_tune_at = len(base_model.layers) - 20
            for i, layer in enumerate(base_model.layers):
                if i < fine_tune_at:
                    layer.trainable = False

            logger.info(f"Fine-tuning from layer {fine_tune_at} onwards")

            # Recompile with lower learning rate
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001 / 10),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            # Setup callbacks for fine-tuning
            callbacks = self.setup_callbacks(model_save_path)

            # Fine-tuning
            history = self.model.fit(
                train_generator,
                epochs=epochs,
                validation_data=val_generator,
                callbacks=callbacks,
                verbose=1
            )

            logger.info("\n=== FINE-TUNING COMPLETED ===")
            return history

        except Exception as e:
            logger.error(f"Error during fine-tuning: {str(e)}")
            raise

    def plot_training_history(
        self,
        history: tf.keras.callbacks.History,
        save_path: str = 'training_history.png'
    ) -> None:
        """
        Plot and save training history.

        Args:
            history: Training history object
            save_path: Path to save the plot
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # Accuracy
            ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
            ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
            ax1.set_title('Model Accuracy', fontsize=14)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Loss
            ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
            ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
            ax2.set_title('Model Loss', fontsize=14)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Training plot saved at: {save_path}")

        except Exception as e:
            logger.error(f"Error plotting training history: {str(e)}")
            raise

    def evaluate_model(
        self,
        test_generator: ImageDataGenerator
    ) -> Optional[Dict[str, float]]:
        """
        Evaluate model on test set.

        Args:
            test_generator: Test data generator

        Returns:
            Dictionary containing evaluation metrics if successful, None otherwise
        """
        if self.model is None:
            logger.error("Model not loaded.")
            return None

        logger.info("\n=== EVALUATING MODEL ===")

        try:
            # Evaluate
            test_loss, test_accuracy = self.model.evaluate(test_generator, verbose=1)

            logger.info(f"\nEvaluation Results:")
            logger.info(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
            logger.info(f"Test Loss: {test_loss:.4f}")

            return {'accuracy': test_accuracy, 'loss': test_loss}

        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise

    def save_model(self, filepath: str) -> None:
        """
        Save the model and its weights.

        Args:
            filepath: Path to save the model
        """
        if self.model is not None:
            try:
                self.model.save(filepath)
                logger.info(f"Model saved at: {filepath}")

                # Save weights separately
                weights_path = filepath.replace('.h5', '_weights.h5')
                self.model.save_weights(weights_path)
                logger.info(f"Model weights saved at: {weights_path}")

            except Exception as e:
                logger.error(f"Error saving model: {str(e)}")
                raise
        else:
            logger.error("No model to save.")


def create_dataset_structure() -> None:
    """Create directory structure for the dataset."""
    folders = [
        'dataset/train/green',
        'dataset/train/ripe',
        'dataset/val/green',
        'dataset/val/ripe',
        'dataset/test/green',
        'dataset/test/ripe'
    ]

    try:
        for folder in folders:
            os.makedirs(folder, exist_ok=True)

        logger.info("Dataset directory structure:")
        logger.info("dataset/")
        logger.info("├── train/")
        logger.info("│   ├── green/  (place green tomato images)")
        logger.info("│   └── ripe/   (place ripe tomato images)")
        logger.info("├── val/")
        logger.info("│   ├── green/")
        logger.info("│   └── ripe/")
        logger.info("└── test/")
        logger.info("    ├── green/")
        logger.info("    └── ripe/")

    except Exception as e:
        logger.error(f"Error creating dataset structure: {str(e)}")
        raise


def main() -> None:
    """Main function to run the training pipeline."""
    parser = argparse.ArgumentParser(description='Training Tomato Classifier')
    parser.add_argument('--train_dir', default='dataset/train', help='Training data directory')
    parser.add_argument('--val_dir', default='dataset/val', help='Validation data directory')
    parser.add_argument('--test_dir', default='dataset/test', help='Test data directory')
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--fine_tune_epochs', type=int, default=10, help='Number of fine-tuning epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--model_name', default='tomato_mobilenetv3', help='Model name for saving')
    parser.add_argument('--create_dirs', action='store_true', help='Create dataset directory structure')

    args = parser.parse_args()

    if args.create_dirs:
        create_dataset_structure()
        return

    # Check GPU availability
    logger.info("GPU Available: %s", tf.config.list_physical_devices('GPU'))

    # Initialize trainer
    trainer = TomatoTrainer()

    try:
        # Prepare data
        train_generator, val_generator = trainer.prepare_data_generators(
            args.train_dir,
            args.val_dir,
            batch_size=args.batch_size
        )

        # Create model
        trainer.create_model()

        # Training
        model_path = f'{args.model_name}.h5'
        history = trainer.train_model(
            train_generator,
            val_generator,
            epochs=args.epochs,
            model_save_path=model_path
        )

        # Plot training history
        trainer.plot_training_history(history, f'{args.model_name}_history.png')

        # Fine-tuning (optional)
        if args.fine_tune_epochs > 0:
            fine_tuned_path = f'{args.model_name}_fine_tuned.h5'
            history_ft = trainer.fine_tune_model(
                train_generator,
                val_generator,
                epochs=args.fine_tune_epochs,
                model_save_path=fine_tuned_path
            )

            # Plot fine-tuning history
            trainer.plot_training_history(history_ft, f'{args.model_name}_fine_tune_history.png')

        # Evaluate on test set (if available)
        if os.path.exists(args.test_dir):
            test_generator, _ = trainer.prepare_data_generators(
                args.test_dir,
                args.test_dir,
                batch_size=args.batch_size
            )
            trainer.evaluate_model(test_generator)

        logger.info(f"\n=== TRAINING COMPLETED ===")
        logger.info(f"Model saved: {model_path}")

    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()