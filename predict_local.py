import os
import sys

def main():
    print("====================================================")
    print("      DEEPFAKE DETECTION LOCAL PREDICTOR            ")
    print("====================================================\n")
    
    # 1. Check for required libraries
    try:
        import tensorflow as tf
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"Error: Missing required library: {e.name}")
        print("Please install the requirements first by running:")
        print("pip install tensorflow opencv-python matplotlib numpy")
        return

    # 2. Look for the trained model file
    model_name = "deepfake_detector.h5"
    downloads_dir = r"C:\Users\SKV\Downloads"
    model_path = os.path.join(downloads_dir, model_name)
    
    if not os.path.exists(model_path):
        # Fallback check in current directory
        if os.path.exists(model_name):
            model_path = model_name
        else:
            print(f"Error: Could not find '{model_name}' in your Downloads folder.")
            print("To run local predictions, you must first:")
            print("  1. Open the corrected notebook in Google Colab.")
            print("  2. Train the model using the free cloud GPU.")
            print("  3. Save the model by running: model.save('deepfake_detector.h5')")
            print(f"  4. Download it and place it in: {downloads_dir}\n")
            return

    # 3. Load the model
    print(f"Loading deep learning model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 4. Get input image path
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = input("Please drag-and-drop or paste the absolute path to your test image:\n> ").strip().strip('\"\'')

    if not img_path:
        print("Error: No image path provided.")
        return

    if not os.path.exists(img_path):
        print(f"Error: File does not exist at '{img_path}'")
        return

    # 5. Load and preprocess image
    print(f"Reading image: {os.path.basename(img_path)}...")
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Could not read image file. Make sure it is a valid JPG/PNG.")
        return

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Preprocess (resize to 224x224, keep raw [0, 255] scale to match fixed EfficientNet)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_batch = np.expand_dims(img_resized, axis=0)

    # 6. Predict
    print("Running deepfake detection inference...")
    try:
        pred_prob = model.predict(img_batch)[0][0]
        
        real_prob = pred_prob * 100
        fake_prob = (1 - pred_prob) * 100
        final_pred = "REAL" if pred_prob > 0.5 else "FAKE"
        
        print("\nPrediction Results:")
        print(f"  Fake Probability : {fake_prob:.2f}%")
        print(f"  Real Probability : {real_prob:.2f}%")
        print(f"  Final Decision   : {final_pred}\n")
        
        # Display image
        plt.figure(figsize=(6, 6))
        plt.imshow(img_rgb)
        plt.title(f"Prediction: {final_pred} (Real: {real_prob:.1f}%, Fake: {fake_prob:.1f}%)")
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"Error running model prediction: {e}")

if __name__ == "__main__":
    main()
