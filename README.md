# 🎵 **Music Genre Classification System**
A comprehensive machine learning and deep learning project that classifies music genres using both **tabular audio features** and **spectrogram-based image data**. This project combines classical machine learning with **CNNs** and **Transfer Learning (VGG16)** for a multi-approach solution to audio classification.

---

📌 **Project Overview**

Classifying music by genre is a common challenge in music information retrieval. This project offers a hybrid approach:

1. **Tabular Audio Features** (e.g., tempo, pitch, spectral contrast) are used with **Random Forests**.
2. **Spectrograms** (visual representation of audio) are processed via:

   * A custom-built **Convolutional Neural Network (CNN)**
   * **Transfer Learning** using **VGG16**, a pre-trained deep model.

By applying multiple models, the system enhances genre prediction accuracy and demonstrates the versatility of different AI approaches.

---

🧠 **Technologies & Libraries Used**

* **Python**
* **Pandas**, **NumPy**
* **Matplotlib**, **Seaborn**
* **Librosa** (for audio processing)
* **Scikit-learn** (for classical ML)
* **TensorFlow / Keras** (for deep learning)
* **VGG16** (pre-trained model from Keras applications)

---

⚙️ **Key Features**

🔸 **Tabular Audio Feature Classification**

* Load tabular `.csv` data with audio features
* Train a **Random Forest Classifier**
* Visualize predictions via **confusion matrix**

🔸 **Spectrogram Generation**

* Automatically convert `.wav` audio files to spectrogram images
* Save and organize spectrograms by genre

🔸 **Custom CNN Model**

* Load and preprocess spectrogram images
* Train a CNN from scratch
* Evaluate accuracy and loss on validation data

🔸 **Transfer Learning with VGG16**

* Leverage ImageNet-trained VGG16 model
* Freeze base layers, train only top classifier
* Improve performance with fewer training resources

---

🖼️ **Visualization Highlights**

* Spectrograms created from raw `.wav` files using **Librosa**
* Heatmaps for **confusion matrices** to visually compare model predictions
* Optional accuracy/loss plots during CNN training

---

📦 **Modular Design**
All functionality is wrapped inside a `MusicGenreClassifier` class, offering clean modularity for:

* Loading data
* Training and evaluating different model types
* Spectrogram generation

---

💻 **Usage Instructions**

1. Install required libraries:

   ```
   pip install pandas numpy matplotlib seaborn scikit-learn librosa tensorflow
   ```

2. Ensure paths are correct for:

   * Tabular data (`features_30_sec.csv`)
   * Audio directory (`/audio`)

3. Run the main script:

   ```
   python your_script_name.py
   ```

---

🔬 **Approach Summary**

| Method              | Input Type       | Model Type                |
| ------------------- | ---------------- | ------------------------- |
| Tabular             | Numeric features | Random Forest             |
| Image (Spectrogram) | Visual/audio     | Custom CNN                |
| Image (Spectrogram) | Visual/audio     | Transfer Learning (VGG16) |

---

This project is an ideal real-world example of combining **machine learning**, **audio signal processing**, and **deep learning** to solve a complex classification problem in the domain of **music analysis** and **AI-powered recommendation systems**.
