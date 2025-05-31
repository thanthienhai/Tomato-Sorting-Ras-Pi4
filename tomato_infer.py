"""
Tomato Classification Inference Script
Predict ripe/green tomatoes using a trained model
"""

import os
import sys
import logging
import subprocess
from typing import Union, Dict, Optional, Tuple, Any
from pathlib import Path
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def install_requirements() -> bool:
    """
    Install required packages from requirements.txt.
    
    Returns:
        bool: True if installation was successful, False otherwise
    """
    try:
        requirements_file = "requirements.txt"
        if not os.path.exists(requirements_file):
            logger.error(f"requirements.txt not found in {os.getcwd()}")
            return False

        logger.info("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        logger.info("All required packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing packages: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during package installation: {str(e)}")
        return False

def check_packages() -> bool:
    """
    Check and verify required packages are installed.
    
    Returns:
        bool: True if all packages are correctly installed, False otherwise
    """
    try:
        # Try importing required packages
        import numpy as np
        import tensorflow as tf
        import cv2
        from PIL import Image
        import matplotlib.pyplot as plt

        # Check versions
        required_versions = {
            'numpy': '1.23.5',
            'tensorflow': '2.10.0',
            'opencv-python-headless': '4.8.0.76',
            'pillow': '10.0.0',
            'matplotlib': '3.7.2'
        }

        missing_packages = []
        version_mismatches = []

        for package, required_version in required_versions.items():
            try:
                if package == 'opencv-python-headless':
                    installed_version = cv2.__version__
                else:
                    installed_version = eval(f"{package}.__version__")

                if installed_version != required_version:
                    version_mismatches.append(f"{package} (required: {required_version}, installed: {installed_version})")
            except (AttributeError, NameError):
                missing_packages.append(package)

        if missing_packages or version_mismatches:
            logger.error("\n=== PACKAGE ISSUES DETECTED ===")
            if missing_packages:
                logger.error("Missing packages:")
                for package in missing_packages:
                    logger.error(f"  - {package}")
            if version_mismatches:
                logger.error("Version mismatches:")
                for mismatch in version_mismatches:
                    logger.error(f"  - {mismatch}")
            
            logger.error("\nInstalling correct versions...")
            if not install_requirements():
                logger.error("Failed to install required packages. Please install them manually:")
                logger.error("pip install -r requirements.txt")
                return False
            
            # Verify installation after attempting to fix
            return check_packages()
        
        return True
            
    except ImportError as e:
        logger.error(f"Error importing required packages: {str(e)}")
        logger.error("Installing required packages...")
        if not install_requirements():
            logger.error("Failed to install required packages. Please install them manually:")
            logger.error("pip install -r requirements.txt")
            return False
        # Verify installation after attempting to fix
        return check_packages()

def setup_environment() -> bool:
    """
    Setup the environment and verify all requirements.
    
    Returns:
        bool: True if setup was successful, False otherwise
    """
    # Check and install packages
    if not check_packages():
        logger.error("Failed to setup environment. Please check the errors above.")
        return False
    
    # Additional environment setup if needed
    try:
        # Configure matplotlib to use non-interactive backend
        import matplotlib
        matplotlib.use('Agg')
        
        # Configure TensorFlow to use CPU only
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        
        return True
    except Exception as e:
        logger.error(f"Error during environment setup: {str(e)}")
        return False

# Setup environment before importing other packages
if not setup_environment():
    sys.exit(1)

# Import required packages
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large

class TomatoInference:
    def __init__(self, model_path: str, img_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the TomatoInference class.

        Args:
            model_path: Path to the trained model file
            img_size: Input image size (height, width)
        """
        self.img_size = img_size
        self.model = None
        self.class_names = ['green', 'ripe']  # 0: green, 1: ripe
        self.class_names_vi = ['Xanh', 'Chín']  # Vietnamese
        
        # Validate model path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.load_model(model_path)

    def load_model(self, model_path: str) -> bool:
        """
        Load a trained model.

        Args:
            model_path: Path to the model file

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading model from: {model_path}")
            
            # Check file extension
            if not model_path.endswith('.h5'):
                raise ValueError("Model file must have .h5 extension")
            
            # Load model with custom_objects if needed
            self.model = tf.keras.models.load_model(
                model_path,
                compile=False,  # Don't compile the model
                custom_objects={'tf': tf}  # Add custom objects if needed
            )
            
            # Verify model structure
            if not isinstance(self.model, tf.keras.Model):
                raise ValueError("Loaded model is not a valid Keras model")
            
            # Compile model with appropriate settings
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("Model loaded successfully!")
            logger.info(f"Input shape: {self.model.input_shape}")
            logger.info(f"Output shape: {self.model.output_shape}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def preprocess_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess image for inference.

        Args:
            image_input: Input image (file path, numpy array, or PIL Image)

        Returns:
            Tuple of (processed_image, original_image)
        """
        try:
            # Handle different input types
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image file not found: {image_input}")
                    
                # Check file extension
                if not image_input.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    raise ValueError("Unsupported image format")
                    
                img = cv2.imread(image_input)
                if img is None:
                    raise ValueError(f"Failed to read image: {image_input}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            elif isinstance(image_input, np.ndarray):
                img = image_input
                if len(img.shape) != 3:
                    raise ValueError("Input image must be 3D (height, width, channels)")
                    
            elif isinstance(image_input, Image.Image):
                img = np.array(image_input)
            else:
                raise ValueError("Invalid input type. Expected file path, numpy array, or PIL Image")

            # Store original image
            original_img = img.copy()

            # Resize to model input size
            img_resized = cv2.resize(img, self.img_size)

            # Normalize to [0, 1] - match training preprocessing
            img_normalized = img_resized.astype(np.float32) / 255.0

            # Add batch dimension
            img_batch = np.expand_dims(img_normalized, axis=0)

            return img_batch, original_img

        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise

    def predict_single_image(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        show_image: bool = True,
        save_result: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Predict class for a single image.

        Args:
            image_input: Input image (file path, numpy array, or PIL Image)
            show_image: Whether to display the image with prediction
            save_result: Whether to save prediction results

        Returns:
            Dictionary containing prediction results if successful, None otherwise
        """
        if self.model is None:
            logger.error("Model not loaded. Please load a model first.")
            return None

        try:
            # Preprocess image
            processed_img, original_img = self.preprocess_image(image_input)

            # Get prediction
            predictions = self.model.predict(processed_img, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]

            # Prepare results
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

            # Log results
            logger.info("\n=== PREDICTION RESULTS ===")
            logger.info(f"Tomato Type: {result['predicted_class_vi']}")
            logger.info(f"Confidence: {result['confidence']:.2%}")
            logger.info("Probabilities:")
            logger.info(f"  - Green: {result['probabilities']['green']:.2%}")
            logger.info(f"  - Ripe: {result['probabilities']['ripe']:.2%}")

            # Display image if requested
            if show_image:
                self.display_prediction(original_img, result)

            # Save results if requested
            if save_result:
                self.save_prediction_result(result, image_input)

            return result

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return None

    def display_prediction(self, image: np.ndarray, result: Dict[str, Any]) -> None:
        """
        Display image with prediction results.

        Args:
            image: Input image
            result: Prediction results
        """
        try:
            plt.figure(figsize=(8, 6))
            plt.imshow(image)

            # Create title with prediction info
            title = f"Prediction: {result['predicted_class_vi']}\n"
            title += f"Confidence: {result['confidence']:.1%}"

            # Set color based on prediction
            color = 'green' if result['predicted_class'] == 'green' else 'red'

            plt.title(title, fontsize=14, color=color, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.show()

        except Exception as e:
            logger.error(f"Error displaying prediction: {str(e)}")
            raise

    def save_prediction_result(self, result: Dict[str, Any], image_path: Union[str, Path]) -> None:
        """
        Save prediction results to JSON file.

        Args:
            result: Prediction results
            image_path: Path to the input image
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prediction_{timestamp}.json"

            # Add image info
            result['image_info'] = {
                'path': str(image_path),
                'name': os.path.basename(str(image_path))
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            logger.info(f"Results saved to: {filename}")

        except Exception as e:
            logger.error(f"Error saving prediction results: {str(e)}")
            raise


def main() -> None:
    """Main function to run inference."""
    # Model and image paths
    MODEL_PATH = "tomato_mobilenetv3_fine_tuned.h5"  # Path to your model file
    IMAGE_PATH = "best4_jpeg.rf.3df643a002bf5a83cd6ef2b3ac572eae.jpg"  # Path to your test image

    try:
        # Check if files exist
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        if not os.path.exists(IMAGE_PATH):
            raise FileNotFoundError(f"Image file not found: {IMAGE_PATH}")

        # Initialize inference engine
        inference = TomatoInference(MODEL_PATH)

        # Run prediction
        result = inference.predict_single_image(
            IMAGE_PATH,
            show_image=True,
            save_result=True
        )

        if result:
            logger.info(f"\nFinal Result: {result['predicted_class_vi']} ({result['confidence']:.2%})")

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise


if __name__ == "__main__":
    main()