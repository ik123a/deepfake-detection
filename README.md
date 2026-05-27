# Premium Deepfake Detection using EfficientNetB0

A state-of-the-art Deepfake Image Detection pipeline built using TensorFlow/Keras and pre-trained EfficientNetB0. This project features high-efficiency transfer learning, robust data pipeline preprocessing (free from the infamous EfficientNet double-rescaling bug), dual runtime support (Google Colab GPU and quick laptop CPU modes), and a standalone desktop prediction client.

---

## 🌟 Key Features

* **EfficientNetB0 Backbone:** Leverages transfer learning with pre-trained ImageNet weights for highly accurate spatial feature extraction.
* **Double-Rescaling Resolution:** Configured to avoid Keras's built-in scaling conflict, allowing gradients to flow correctly and achieving high validation accuracy rapidly.
* **Smart Laptop Mode (`LAPTOP_MODE = True`):** Enables rapid pipeline verification on standard CPUs (takes only ~3 minutes) before kicking off heavy cloud training.
* **Interactive Visualizations:** Renders live training accuracy/loss curves and clean, correctly labeled confusion matrices to evaluate True Positives and False Positives accurately.
* **Robust Local Predictor:** Includes a standalone desktop script to load trained model weights (`.h5`) and run instant predictions on any custom image locally.

---

## 🛠️ Project Structure

```
├── DFDM_fixed.ipynb     # The core Jupyter Notebook for training & evaluation (Colab/Local)
├── predict_local.py     # Standalone local prediction CLI client
└── README.md            # Project documentation and setup guide
```

---

## 🚀 Quick Start Guide

### 1. Environment Installation
Ensure you have Python 3.10+ installed. Open your terminal or Conda command prompt and run:
```bash
pip install tensorflow opencv-python matplotlib scikit-learn kagglehub numpy
```

### 2. Model Training & Fine-Tuning
Open `DFDM_fixed.ipynb` in your environment (Jupyter Lab, VS Code, or Google Colab):
* **On Laptop CPU (Lightweight Testing):** Set `LAPTOP_MODE = True` in Cell 4. The notebook will automatically train on a small, fast subset of the data, verify the entire pipeline, and save your model to your Downloads folder in under 3 minutes.
* **On Google Colab (Full Training):** Set `LAPTOP_MODE = False`, enable the **T4 GPU** runtime, and run all cells. The pipeline will train on the full 112,000 images in under an hour, yielding highly accurate deepfake detection. Save the resulting weights using `model.save("deepfake_detector.h5")`.

---

## 🔍 Standalone Local Predictions

Once you have trained the model and have the `deepfake_detector.h5` weights file in your Downloads folder, you can run instant local predictions on any image:

1. Open your terminal and run:
   ```bash
   python predict_local.py
   ```
2. Paste or drag-and-drop the path to your image when prompted:
   ```text
   Please paste the absolute path to your test image:
   > C:\Users\SKV\Pictures\my_photo.jpg
   ```
3. The script will instantly output the deepfake probability and render the decision:
   ```text
   Prediction Results:
     Fake Probability : 98.42%
     Real Probability : 1.58%
     Final Decision   : FAKE
   ```

---

## 📊 Model Architecture details

* **Base Network:** EfficientNetB0 (Feature extractor, frozen during Stage 1, fine-tuned in Stage 2).
* **Global Average Pooling:** Converts 2D feature maps into a 1D vector.
* **Fully Connected Layer:** 256 Neurons with ReLU activation.
* **Regularization:** Dropout layer (rate = 0.5) to prevent overfitting.
* **Output Classification:** 1 Neuron with Sigmoid activation (outputs class `0` for Fake and `1` for Real).
