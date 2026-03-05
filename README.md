# Underwater Acoustic Vessel Classification System

## Project Overview
This project aims to build an end-to-end binary audio classification system that determines whether underwater acoustic recordings contain vessel sounds (*e.g., ships, boats, propellers, UUVs*) or not (non-vessel sounds, such as marine animals, natural ocean sounds, or background noise). This system serves as a foundational component for maritime domain awareness and acoustic monitoring.

## Design Constraints
The system is engineered with several key constraints to ensure practicality and extensibility:
*   **Offline Operation**: Must function without requiring an internet connection.
*   **CPU-Compatible**: Designed to run efficiently on standard CPU hardware, avoiding reliance on GPUs.
*   **Classical Machine Learning**: Utilizes traditional ML algorithms (specifically SVM), promoting interpretability and lower computational overhead.
*   **Modular and Production-Ready Architecture**: Follows a structured, maintainable design suitable for deployment and future enhancements.
*   **Extensibility**: Built to easily accommodate future stages, such as multi-class classification (e.g., different types of vessels) and specific event detection.

## Overall System Pipeline
The system operates through a robust, end-to-end machine learning pipeline:

1.  **Raw Audio Loading**: Ingests raw `.wav` audio files.
2.  **Audio Preprocessing**: Converts to mono, resamples to a uniform sample rate (16 kHz), and normalizes amplitude.
3.  **Segmentation**: Divides preprocessed audio into fixed-length segments, discarding silent or low-energy parts.
4.  **Feature Extraction**: Derives meaningful numerical features (MFCCs) from each valid audio segment.
5.  **Dataset Preparation**: Aggregates features and labels into a structured dataset.
6.  **Model Training**: Trains a classical machine learning model (SVM) on the prepared dataset.
7.  **Model Evaluation**: Assesses the trained model's performance using standard metrics.
8.  **Inference**: Applies the learned model to new, unseen audio to predict the presence of vessel sounds.

## Detailed Development Steps

The project follows a structured development approach, encompassing environment setup, modular components, and documentation.

### 1. Environment Setup
The system relies on a set of standard Python libraries:
*   `soundfile`: For robust audio file I/O, handling varying bit depths.
*   `librosa`: Core library for audio analysis, including resampling, normalization, and MFCC extraction.
*   `numpy`: Fundamental package for numerical operations and array manipulation.
*   `scikit-learn`: Provides machine learning tools like `StandardScaler`, `SVC` (SVM), and evaluation metrics.
*   `scipy`: For scientific computing, used here for statistical operations (e.g., majority voting in inference).
*   `pandas`: For data manipulation and structured datasets.
*   `joblib`: For efficient serialization (saving/loading) of Python objects, specifically models and scalers.
*   `python-docx`: Used for reading project requirements from `.docx` files.

### 2. Project Structure

The project adheres to a clear, hierarchical directory structure:
```
vessel_detection/
├── data/               # Stores raw and processed datasets
├── src/                # Source code modules
├── models/             # Stores trained ML models and feature scalers
├── requirements.txt    # Python dependencies
├── main.py             # The main entry point for training and inference operations
└── README.md           # Project documentation, setup, and usage guide
```

### 3. Modular Components (`src/` directory)

The `src` directory houses the core logic, divided into several specialized modules:

*   **`audio_processor.py`**: Contains functions for loading, preprocessing (mono conversion, resampling to 16 kHz, normalization), and segmenting audio files (2-second windows, Root Mean Square (RMS) energy thresholding to remove silence).

*   **`feature_extractor.py`**: Implements the MFCC feature extraction, computing mean and standard deviation for 13 coefficients to form a 26-dimensional feature vector from each segment.

*   **`prepare_dataset.py`**: Orchestrates data processing, iterating through `vessel/` and `non_vessel/` directories, applying `audio_processor` and `feature_extractor` functions, and saving compiled features and labels as NumPy arrays (`features.npy`, `labels.npy`).

*   **`train_model.py`**: Handles loading the prepared dataset, splitting data into 80/20 train/test sets, standardizing features with `StandardScaler`, training an SVM with an RBF kernel, and saving both the scaler and the trained model (`scaler.joblib`, `svm_model.joblib`).

*   **`evaluate_model.py`**: Loads the trained model and scaler, scales test features, makes predictions, and reports accuracy, precision, recall, and the confusion matrix.

*   **`inference.py`**: Provides functionality to apply the complete pipeline (preprocessing, segmentation, feature extraction, scaling, prediction) to new, unseen audio files, aggregating segment predictions via majority voting.

### 4. Main Entry Point (`main.py`)

The `main.py` script acts as the command-line interface for the system, utilizing `argparse` to handle user commands:
*   **`python main.py train`**: Executes the full training pipeline, including dataset preparation, model training, and evaluation.
*   **`python main.py infer --audio_file /path/to/audio.wav`**: Performs inference on a specified audio file, outputting the predicted class.

### 5. Machine Learning Model

The chosen machine learning model is a **Support Vector Machine (SVM)** with a **Radial Basis Function (RBF) kernel**. Prior to training the SVM, all features are standardized using `StandardScaler` to ensure that each feature contributes equally to the distance calculation and to prevent features with larger numerical ranges from dominating the learning process.

### 6. Dataset Preprocessing Breakdown

The preprocessing pipeline for audio data is meticulous:

*   **Audio Loading**: Audio files are loaded using `soundfile`, ensuring robustness against varying bit depths and handling potential errors for corrupted files by skipping them.
*   **Mono Conversion**: Stereo audio streams are converted to single-channel (mono) to simplify processing and reduce dimensionality.
*   **Resampling**: All audio is uniformly resampled to a target sample rate of **16 kHz** to ensure consistency across the dataset.
*   **Normalization**: Audio amplitudes are normalized to a standard range (e.g., -1 to 1) to prevent clipping and improve model stability.
*   **Segmentation with RMS Thresholding**: Preprocessed audio is segmented into **2-second, non-overlapping windows**. For each segment, the Root Mean Square (RMS) energy is calculated using `librosa.feature.rms`. Segments falling below a predefined RMS threshold are discarded, effectively removing periods of silence or background noise.
*   **MFCC Feature Extraction**: From each valid audio segment, **13 Mel-Frequency Cepstral Coefficients (MFCCs)** are extracted. To capture the spectral characteristics over time, the **mean and standard deviation** for each of these 13 coefficients are computed across the segment. This results in a **26-dimensional feature vector** (13 means + 13 standard deviations) for every 2-second segment.
*   **Standardization**: The extracted 26-dimensional feature vectors are then standardized using a `StandardScaler`, which transforms the data to have a mean of 0 and a standard deviation of 1. This step is crucial for SVMs to perform optimally.

### 7. Training Strategy

The dataset (composed of extracted and standardized MFCC feature vectors and their corresponding labels) is split into an **80% training set** and a **20% testing set**. This split is performed using a **stratified approach** (`stratify=y` in `train_test_split`) to ensure that the proportion of vessel and non-vessel samples is maintained in both the training and testing sets, which is important for imbalanced datasets. The `StandardScaler` is fitted *only* on the training data to prevent data leakage.

### 8. Evaluation Metrics

Model performance is comprehensively evaluated using the following metrics:
*   **Accuracy**: The proportion of correctly classified samples.
*   **Precision**: The ability of the model to correctly identify positive samples (vessels).
*   **Recall**: The ability of the model to find all positive samples (vessels).
*   **Confusion Matrix**: A table showing the number of true positives, true negatives, false positives, and false negatives, providing a detailed breakdown of classification performance.

### 9. Inference Mechanism

For inference on new, unseen audio files, the system mimics the training pipeline's preprocessing steps. The new audio is loaded, preprocessed, segmented, and features are extracted and scaled. Each 2-second segment then receives an individual prediction from the trained SVM model. To produce a single, conclusive classification for the entire audio file, these segment-level predictions are aggregated using **majority voting**. This ensures robustness against transient noise or intermittent signals within the audio.

## Data Requirements and Setup

To run this project, you must provide your `.wav` audio dataset files and organize them in the following directory structure within the `vessel_detection/data/` folder:

*   **`vessel_detection/data/vessel/`**: Place all `.wav` files containing vessel sounds here.
*   **`vessel_detection/data/non_vessel/`**: Place all `.wav` files containing non-vessel sounds here.

### Example: Copying Data from Google Drive (if applicable)

If your audio files are stored in Google Drive, you can mount your Drive and copy them to the project's data directories. First, mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then, copy your files. Remember to replace `My_Vessel_Sounds_Folder` and `My_Non_Vessel_Sounds_Folder` with the actual paths to your folders in Google Drive (e.g., `/content/drive/My Drive/My_Vessel_Sounds_Folder`).

```python
import os
import shutil

drive_vessel_path = '/content/drive/My Drive/My_Vessel_Sounds_Folder' # <--- UPDATE THIS PATH
drive_non_vessel_path = '/content/drive/My Drive/My_Non_Vessel_Sounds_Folder' # <--- UPDATE THIS PATH

local_vessel_data_path = 'vessel_detection/data/vessel'
local_non_vessel_data_path = 'vessel_detection/data/non_vessel'

os.makedirs(local_vessel_data_path, exist_ok=True)
os.makedirs(local_non_vessel_data_path, exist_ok=True)

def copy_wav_files(source_dir, destination_dir):
    if not os.path.exists(source_dir):
        print(f"Source directory not found: {source_dir}")
        return
    for filename in os.listdir(source_dir):
        if filename.endswith('.wav'):
            source_file = os.path.join(source_dir, filename)
            destination_file = os.path.join(destination_dir, filename)
            try:
                shutil.copy2(source_file, destination_file)
                print(f"Copied {filename} to {destination_dir}")
            except Exception as e:
                print(f"Error copying {filename}: {e}")

print(f"Copying vessel files from '{drive_vessel_path}' to '{local_vessel_data_path}'...")
copy_wav_files(drive_vessel_path, local_vessel_data_path)

print(f"
Copying non-vessel files from '{drive_non_vessel_path}' to '{local_non_vessel_data_path}'...")
copy_wav_files(drive_non_vessel_path, local_non_vessel_data_path)

print("
Data transfer complete. You can now proceed with training the model.")
```

### Inference Audio Files Location
For inference on new, unknown audio files, create a directory `vessel_detection/validation_data/` and place your `.wav` files there.

## Setup Instructions
1.  **Clone the repository** (if applicable).
2.  **Install dependencies**: All required Python packages are listed in `requirements.txt`. You can install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Model
After placing your `.wav` dataset files in `vessel_detection/data/vessel/` and `vessel_detection/data/non_vessel/`, initiate the training process by running the `main.py` script:

```bash
python main.py train
```
This command will prepare the dataset, train the SVM model, and evaluate its performance, saving the trained model and scaler to `vessel_detection/models/`.

### Performing Inference
To perform inference on a new audio file, use the `infer` mode and provide the path to your audio file using the `--audio_file` argument. For example, if you have `my_new_audio.wav` in `vessel_detection/validation_data/`:

```bash
python main.py infer --audio_file vessel_detection/validation_data/my_new_audio.wav
```
Replace `/path/to/your/audio.wav` with the actual path to the `.wav` file you want to classify.

