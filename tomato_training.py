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


class TomatoTrainer:
    def __init__(self, img_size=(224, 224), num_classes=2):
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.class_names = ['green', 'ripe']  # 0: xanh, 1: chín

    def create_model(self):
        """Tạo model sử dụng MobileNet-v3 Large với transfer learning"""
        print("Đang tạo model MobileNet-v3...")

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
        print("\nModel Architecture:")
        self.model.summary()

        return self.model

    def prepare_data_generators(self, train_dir, val_dir, batch_size=32):
        """Chuẩn bị data generators cho training và validation"""
        print(f"Chuẩn bị dữ liệu từ {train_dir} và {val_dir}...")

        # Data augmentation cho training
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

        # Chỉ rescale cho validation
        val_datagen = ImageDataGenerator(rescale=1. / 255)

        # Tạo generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            classes=['green', 'ripe'],
            shuffle=True
        )

        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            classes=['green', 'ripe'],
            shuffle=False
        )

        print(f"Số lượng ảnh training: {train_generator.samples}")
        print(f"Số lượng ảnh validation: {val_generator.samples}")
        print(f"Classes: {train_generator.class_indices}")

        return train_generator, val_generator

    def setup_callbacks(self, model_save_path):
        """Thiết lập callbacks cho training"""
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
                verbose=1,
                save_format='h5'
            ),
            tf.keras.callbacks.CSVLogger(
                f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            )
        ]
        return callbacks

    def train_model(self, train_generator, val_generator, epochs=25, model_save_path='best_tomato_model.h5'):
        """Huấn luyện model"""
        if self.model is None:
            self.create_model()

        print(f"\n=== BẮT ĐẦU TRAINING ===")
        print(f"Epochs: {epochs}")
        print(f"Model sẽ được lưu tại: {model_save_path}")

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

        print("\n=== TRAINING HOÀN THÀNH ===")
        return history

    def fine_tune_model(self, train_generator, val_generator, epochs=10, model_save_path='fine_tuned_tomato_model.h5'):
        """Fine-tuning model bằng cách unfreeze một số layers cuối"""
        if self.model is None:
            print("Model chưa được tạo. Vui lòng train model trước.")
            return None

        print("\n=== BẮT ĐẦU FINE-TUNING ===")

        # Unfreeze top layers của base model
        base_model = self.model.layers[0]
        base_model.trainable = True

        # Freeze các layers đầu, chỉ fine-tune các layers cuối
        fine_tune_at = len(base_model.layers) - 20
        for i, layer in enumerate(base_model.layers):
            if i < fine_tune_at:
                layer.trainable = False

        print(f"Fine-tuning từ layer {fine_tune_at} trở đi")

        # Compile lại với learning rate thấp hơn
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001 / 10),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Setup callbacks cho fine-tuning
        callbacks = self.setup_callbacks(model_save_path)

        # Fine-tuning
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )

        print("\n=== FINE-TUNING HOÀN THÀNH ===")
        return history

    def plot_training_history(self, history, save_path='training_history.png'):
        """Vẽ và lưu biểu đồ quá trình training"""
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
        plt.show()
        print(f"Biểu đồ training đã được lưu tại: {save_path}")

    def evaluate_model(self, test_generator):
        """Đánh giá model trên test set"""
        if self.model is None:
            print("Model chưa được load.")
            return None

        print("\n=== ĐÁNH GIÁ MODEL ===")

        # Evaluate
        test_loss, test_accuracy = self.model.evaluate(test_generator, verbose=1)

        print(f"\nKết quả đánh giá:")
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
        print(f"Test Loss: {test_loss:.4f}")

        return {'accuracy': test_accuracy, 'loss': test_loss}

    def save_model(self, filepath):
        """Lưu model"""
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model đã được lưu tại: {filepath}")

            # Lưu thêm chỉ weights
            weights_path = filepath.replace('.h5', '_weights.h5')
            self.model.save_weights(weights_path)
            print(f"Model weights đã được lưu tại: {weights_path}")
        else:
            print("Không có model nào để lưu.")


def create_dataset_structure():
    """Tạo cấu trúc thư mục cho dataset"""
    folders = [
        'dataset/train/green',
        'dataset/train/ripe',
        'dataset/val/green',
        'dataset/val/ripe',
        'dataset/test/green',
        'dataset/test/ripe'
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    print("Cấu trúc thư mục dataset:")
    print("dataset/")
    print("├── train/")
    print("│   ├── green/  (đặt ảnh cà chua xanh)")
    print("│   └── ripe/   (đặt ảnh cà chua chín)")
    print("├── val/")
    print("│   ├── green/")
    print("│   └── ripe/")
    print("└── test/")
    print("    ├── green/")
    print("    └── ripe/")


def main():
    parser = argparse.ArgumentParser(description='Training Tomato Classifier')
    parser.add_argument('--train_dir', default='dataset/train', help='Thư mục training data')
    parser.add_argument('--val_dir', default='dataset/val', help='Thư mục validation data')
    parser.add_argument('--test_dir', default='dataset/test', help='Thư mục test data')
    parser.add_argument('--epochs', type=int, default=25, help='Số epochs cho training')
    parser.add_argument('--fine_tune_epochs', type=int, default=10, help='Số epochs cho fine-tuning')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--model_name', default='tomato_mobilenetv3', help='Tên model để lưu')
    parser.add_argument('--create_dirs', action='store_true', help='Tạo cấu trúc thư mục dataset')

    args = parser.parse_args()

    if args.create_dirs:
        create_dataset_structure()
        return

    # Kiểm tra GPU
    print("GPU Available:", tf.config.list_physical_devices('GPU'))

    # Khởi tạo trainer
    trainer = TomatoTrainer()

    try:
        # Chuẩn bị dữ liệu
        train_generator, val_generator = trainer.prepare_data_generators(
            args.train_dir,
            args.val_dir,
            batch_size=args.batch_size
        )

        # Tạo model
        trainer.create_model()

        # Training
        model_path = f'{args.model_name}.h5'
        history = trainer.train_model(
            train_generator,
            val_generator,
            epochs=args.epochs,
            model_save_path=model_path
        )

        # Vẽ biểu đồ training
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

            # Vẽ biểu đồ fine-tuning
            trainer.plot_training_history(history_ft, f'{args.model_name}_fine_tune_history.png')

        # Đánh giá trên test set (nếu có)
        if os.path.exists(args.test_dir):
            test_generator, _ = trainer.prepare_data_generators(
                args.test_dir,
                args.test_dir,
                batch_size=args.batch_size
            )
            trainer.evaluate_model(test_generator)

        print(f"\n=== TRAINING HOÀN TẤT ===")
        print(f"Model đã được lưu: {model_path}")

    except Exception as e:
        print(f"Lỗi trong quá trình training: {e}")


if __name__ == "__main__":
    main()