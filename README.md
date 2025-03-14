# Image-Search

# Image Feature Extraction and Retrieval System

An advanced system for extracting and comparing image features using traditional computer vision techniques.

## 📌 Overview
This project implements an **Image Feature Extraction and Retrieval System** designed to process and analyze images using traditional feature extraction techniques. It leverages methods such as **Color Histograms, Local Binary Patterns (LBP), and Discrete Cosine Transform (DCT)** to extract meaningful features from images, followed by **dimensionality reduction using PCA** and **similarity measurement for retrieval tasks**. The system evaluates performance using **precision-recall curves**, making it suitable for tasks like image search, classification, or clustering.

Built with **Python** and popular libraries like **OpenCV, scikit-image, and TensorFlow**, this project showcases a robust pipeline for image analysis and retrieval.

## 🚀 Features
### 🔹 Feature Extraction
- **Color Histograms**: Captures color distribution in images.
- **Local Binary Patterns (LBP)**: Extracts texture-based features.
- **Discrete Cosine Transform (DCT)**: Analyzes frequency-based image features.
- **Dimensionality Reduction**: PCA to reduce feature vectors for efficient computation.
- **Similarity Measurement**: Computes distances between feature vectors for image retrieval.
- **Evaluation**: Precision-recall curves to assess retrieval performance.
- **Modular Design**: Easy-to-extend code structure for adding new features or datasets.

## 📂 Dataset
This project uses the **DTD (Describable Textures Dataset)**, a collection of texture images split into training and testing sets. The dataset is loaded from text files (**train1.txt** and **test1.txt**) containing image paths.
- **Training Set**: Used to build the feature database.
- **Testing Set**: Used to evaluate retrieval performance.

**Note**: Replace the placeholder paths in the code with your dataset location.

## 🔧 Requirements
To run this project, ensure you have the following dependencies installed:
```sh
pip install -r requirements.txt
```
### 📦 Key Libraries
- **opencv-python (cv2)**: Image processing and feature extraction.
- **numpy**: Numerical operations.
- **matplotlib**: Plotting precision-recall curves.
- **scikit-image**: LBP feature extraction.
- **scipy**: DCT and distance calculations.
- **sklearn**: PCA and other utilities.
- **tensorflow** (optional): For deep feature extraction (if extended).

## 💾 Installation
### 1️⃣ Clone the Repository:
```sh
git clone https://github.com/yourusername/image-feature-extraction.git
cd image-feature-extraction
```
### 2️⃣ Install Dependencies:
```sh
pip install -r requirements.txt
```
### 3️⃣ Download the Dataset:
- Download the **DTD dataset** from [Oxford Visual Geometry Group](https://www.robots.ox.ac.uk/~vgg/data/dtd/).
- Extract it and update the paths in the code (e.g., `C:\Users\M\Downloads\dtd-r1.0.1\dtd\images`).

### 4️⃣ Run the Project:
```sh
jupyter notebook image_feature_extraction.ipynb
```

## 🎯 Usage
### ✅ Load the Dataset:
- The code reads image paths from **train1.txt** and **test1.txt** and loads images using OpenCV.

### ✅ Extract Features:
- Run cells to extract **Color Histograms, LBP, and DCT** features from training images.

### ✅ Evaluate Retrieval:
- Compute distances between test and training features.
- Generate **precision-recall curves** to evaluate performance.

### ✅ Visualize Results:
- Precision-recall plots are displayed using **Matplotlib**.

## 📂 Project Structure
```
image-feature-extraction/
│
├── image_feature_extraction.ipynb  # Main Jupyter Notebook
├── requirements.txt                # Dependencies
├── README.md                       # Project documentation
├── data/                           # Dataset folder (add your dataset here)
│   ├── train1.txt                  # Training image paths
│   └── test1.txt                   # Testing image paths
└── images/                         # Sample output plots (optional)
```

## 📊 Results
The system generates **precision-recall curves** for deep features and combined features, as shown below:

### 🔹 **Deep Features Precision-Recall:**
```sh
Precisions: [1.0, 1.0, 0.666..., ...]
Recalls: [0.025, 0.05, 0.05, ...]
```
### 🔹 **Combined Features Precision-Recall:**
```sh
Precisions: [1.0, 1.0, 0.666..., ...]
Recalls: [0.025, 0.05, 0.05, ...]
```
These metrics demonstrate the system's ability to **retrieve relevant images effectively**.

## 🔮 Future Enhancements
- **Add deep learning-based feature extraction** (e.g., CNNs with pre-trained models like VGG or ResNet).
- **Implement a GUI** for interactive image retrieval.
- **Support additional datasets** beyond DTD.
- **Optimize performance** with parallel processing or GPU acceleration.

## 🤝 Contributing
Contributions are welcome! To contribute:
1. **Fork** the repository.
2. Create a feature branch:
   ```sh
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```sh
   git commit -m "Add your feature"
   ```
4. Push to the branch:
   ```sh
   git push origin feature/your-feature
   ```
5. Open a **Pull Request**.

## 📜 License
This project is licensed under the **MIT License**. See the LICENSE file for details.

## 🙌 Acknowledgments
- **DTD Dataset**: Provided by the [Oxford Visual Geometry Group](https://www.robots.ox.ac.uk/~vgg/data/dtd/).
- **Libraries**: OpenCV, scikit-image, TensorFlow, and more.
- **Inspiration**: Traditional computer vision techniques for image analysis.
