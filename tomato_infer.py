"""
Tomato Classification Inference Script
Dự đoán cà chua chín/xanh sử dụng model đã được huấn luyện
"""

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os
import json
from datetime import datetime
import glob


class TomatoInference:
    def __init__(self, model_path=None, img_size=(224, 224)):
        self.img_size = img_size
        self.model = None
        self.class_names = ['green', 'ripe']  # 0: xanh, 1: chín
        self.class_names_vi = ['Xanh', 'Chín']  # Tiếng Việt

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        """Load model đã được huấn luyện"""
        try:
            print(f"Đang load model từ: {model_path}")
            self.model = tf.keras.models.load_model(model_path)
            print("Model đã được load thành công!")
            print(f"Input shape: {self.model.input_shape}")
            print(f"Output shape: {self.model.output_shape}")
        except Exception as e:
            print(f"Lỗi khi load model: {e}")
            return False
        return True

    def preprocess_image(self, image_input):
        """Tiền xử lý ảnh cho inference"""
        # Xử lý input (có thể là file path, numpy array, hoặc PIL Image)
        if isinstance(image_input, str):
            # Load từ file path
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Không tìm thấy file: {image_input}")
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Không thể đọc ảnh: {image_input}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, np.ndarray):
            img = image_input
        elif isinstance(image_input, Image.Image):
            img = np.array(image_input)
        else:
            raise ValueError("Input không hợp lệ. Cần file path, numpy array, hoặc PIL Image")

        # Resize về kích thước model yêu cầu
        img_resized = cv2.resize(img, self.img_size)

        # Normalize về [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0

        # Thêm batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)

        return img_batch, img

    def predict_single_image(self, image_input, show_image=True, save_result=False):
        """Dự đoán cho một ảnh"""
        if self.model is None:
            print("Model chưa được load. Vui lòng load model trước.")
            return None

        try:
            # Preprocess
            processed_img, original_img = self.preprocess_image(image_input)

            # Prediction
            predictions = self.model.predict(processed_img, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]

            # Kết quả
            result = {
                'timestamp': datetime.now().isoformat(),
                'predicted_class_id': int(predicted_class),
                'predicted_class': self.class_names[predicted_class],
                'predicted_class_vi': self.class_names_vi[predicted_class],
                'confidence': float(confidence),
                'probabilities': {
                    'green': float(predictions[0][0]),
                    'ripe': float(predictions[0][1])
                },
                'probabilities_vi': {
                    'xanh': float(predictions[0][0]),
                    'chin': float(predictions[0][1])
                }
            }

            # Hiển thị kết quả
            print(f"\n=== KẾT QUẢ DỰ ĐOÁN ===")
            print(f"Loại cà chua: {result['predicted_class_vi']}")
            print(f"Độ tin cậy: {result['confidence']:.2%}")
            print(f"Xác suất:")
            print(f"  - Xanh: {result['probabilities']['green']:.2%}")
            print(f"  - Chín: {result['probabilities']['ripe']:.2%}")

            # Hiển thị ảnh
            if show_image:
                self.display_prediction(original_img, result)

            # Lưu kết quả
            if save_result:
                self.save_prediction_result(result, image_input)

            return result

        except Exception as e:
            print(f"Lỗi trong quá trình dự đoán: {e}")
            return None

    def predict_batch(self, image_paths, save_results=False):
        """Dự đoán cho nhiều ảnh"""
        if not image_paths:
            print("Danh sách ảnh trống.")
            return []

        print(f"Đang dự đoán cho {len(image_paths)} ảnh...")
        results = []

        for i, img_path in enumerate(image_paths):
            print(f"\nXử lý ảnh {i + 1}/{len(image_paths)}: {os.path.basename(img_path)}")

            result = self.predict_single_image(
                img_path,
                show_image=False,
                save_result=False
            )

            if result:
                result['image_path'] = img_path
                result['image_name'] = os.path.basename(img_path)
                results.append(result)

        # Lưu kết quả batch
        if save_results and results:
            self.save_batch_results(results)

        # Tóm tắt kết quả
        self.print_batch_summary(results)

        return results

    def predict_folder(self, folder_path, image_extensions=None, save_results=False):
        """Dự đoán cho tất cả ảnh trong thư mục"""
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        # Tìm tất cả file ảnh
        image_paths = []
        for ext in image_extensions:
            pattern = os.path.join(folder_path, f"*{ext}")
            image_paths.extend(glob.glob(pattern, recursive=False))
            pattern = os.path.join(folder_path, f"*{ext.upper()}")
            image_paths.extend(glob.glob(pattern, recursive=False))

        if not image_paths:
            print(f"Không tìm thấy ảnh nào trong thư mục: {folder_path}")
            return []

        print(f"Tìm thấy {len(image_paths)} ảnh trong thư mục: {folder_path}")

        return self.predict_batch(image_paths, save_results=save_results)

    def display_prediction(self, image, result):
        """Hiển thị ảnh với kết quả dự đoán"""
        plt.figure(figsize=(8, 6))
        plt.imshow(image)

        # Tạo title với thông tin dự đoán
        title = f"Dự đoán: {result['predicted_class_vi']}\n"
        title += f"Độ tin cậy: {result['confidence']:.1%}"

        # Màu sắc dựa trên kết quả
        color = 'green' if result['predicted_class'] == 'green' else 'red'

        plt.title(title, fontsize=14, color=color, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def save_prediction_result(self, result, image_path):
        """Lưu kết quả dự đoán đơn lẻ"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"prediction_{timestamp}.json"

        # Thêm thông tin ảnh
        result['image_info'] = {
            'path': str(image_path),
            'name': os.path.basename(str(image_path))
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"Kết quả đã được lưu: {filename}")

    def save_batch_results(self, results):
        """Lưu kết quả dự đoán batch"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_predictions_{timestamp}.json"

        # Tổng hợp thống kê
        summary = {
            'total_images': len(results),
            'green_count': sum(1 for r in results if r['predicted_class'] == 'green'),
            'ripe_count': sum(1 for r in results if r['predicted_class'] == 'ripe'),
            'avg_confidence': np.mean([r['confidence'] for r in results]),
            'timestamp': datetime.now().isoformat()
        }

        batch_data = {
            'summary': summary,
            'predictions': results
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(batch_data, f, ensure_ascii=False, indent=2)

        print(f"Kết quả batch đã được lưu: {filename}")

    def print_batch_summary(self, results):
        """In tóm tắt kết quả batch"""
        if not results:
            return

        green_count = sum(1 for r in results if r['predicted_class'] == 'green')
        ripe_count = sum(1 for r in results if r['predicted_class'] == 'ripe')
        avg_confidence = np.mean([r['confidence'] for r in results])

        print(f"\n=== TÓM TẮT KẾT QUẢ ===")
        print(f"Tổng số ảnh: {len(results)}")
        print(f"Cà chua xanh: {green_count} ({green_count / len(results) * 100:.1f}%)")
        print(f"Cà chua chín: {ripe_count} ({ripe_count / len(results) * 100:.1f}%)")
        print(f"Độ tin cậy trung bình: {avg_confidence:.2%}")

    def create_confidence_histogram(self, results, save_path=None):
        """Tạo biểu đồ phân bố độ tin cậy"""
        if not results:
            return

        confidences = [r['confidence'] for r in results]

        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Độ tin cậy')
        plt.ylabel('Số lượng')
        plt.title('Phân bố độ tin cậy dự đoán')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Biểu đồ đã được lưu: {save_path}")

        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Tomato Classification Inference')
    parser.add_argument('--model', required=True, help='Đường dẫn đến model file (.h5)')
    parser.add_argument('--image', help='Đường dẫn đến ảnh cần dự đoán')
    parser.add_argument('--folder', help='Đường dẫn đến thư mục chứa ảnh')
    parser.add_argument('--batch', nargs='+', help='Danh sách đường dẫn ảnh')
    parser.add_argument('--save_results', action='store_true', help='Lưu kết quả dự đoán')
    parser.add_argument('--no_display', action='store_true', help='Không hiển thị ảnh')

    args = parser.parse_args()

    # Kiểm tra GPU
    print("GPU Available:", tf.config.list_physical_devices('GPU'))

    # Khởi tạo inference engine
    inference = TomatoInference()

    # Load model
    if not inference.load_model(args.model):
        return

    try:
        # Dự đoán cho ảnh đơn lẻ
        if args.image:
            result = inference.predict_single_image(
                args.image,
                show_image=not args.no_display,
                save_result=args.save_results
            )
            if result:
                print(f"\nKết quả: {result['predicted_class_vi']} ({result['confidence']:.2%})")

        # Dự đoán cho thư mục
        elif args.folder:
            results = inference.predict_folder(
                args.folder,
                save_results=args.save_results
            )

            # Tạo biểu đồ nếu có nhiều kết quả
            if len(results) > 5:
                inference.create_confidence_histogram(
                    results,
                    f"confidence_histogram_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )

        # Dự đoán cho batch ảnh
        elif args.batch:
            results = inference.predict_batch(
                args.batch,
                save_results=args.save_results
            )

        else:
            print("Vui lòng chỉ định ảnh (--image), thư mục (--folder), hoặc batch (--batch)")

    except Exception as e:
        print(f"Lỗi trong quá trình inference: {e}")


if __name__ == "__main__":
    main()