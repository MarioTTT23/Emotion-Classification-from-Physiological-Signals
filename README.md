# Emotion-Classification-from-Physiological-Signals
This repository contains a complete Python pipeline for classifying human emotions from physiological signals collected with an **Empatica EmbracePlus** wearable. The project handles the entire workflow from raw `.avro` data to model training, including signal preprocessing, artifact removal, feature extraction, and classification.

Here is a brief overview of the files in this project:

* `classifier.py`: The main executable script. It runs the end-to-end pipeline, from loading data to training the classifier and performing feature selection.
* `Participant.py`: A crucial helper class that encapsulates all data for a single subject, including demographics, questionnaire scores (TAS, PANAS), and experimental markers.
* `params.py`: **Global configuration file.** Stores the file paths for raw data and processed data.
* `experiment_params.py`: Defines all parameters related to the experimental design.
* `load_data.py`: Contains functions to read and parse the raw `.avro` files from the Empatica EmbracePlus.
* `preprocessing.py`: Contains all signal processing functions for cleaning and filtering BVP, EDA, and SKT signals.
* `segmentation.py`: Includes functions to find and extract valid, uncorrupted segments of data for analysis.
* `feature_extraction.py`: Contains functions for calculating all physiological features (HRV, EDA, SKT) from the processed segments.
* `helper.py`: Utility functions that assist in the main pipeline, such as extracting features for all segments of a participant.
---
The physiological data was collected as part of an experiment designed to elicit specific target emotions: **Happiness, Sadness, and Anger**.

In addition to the physiological signals, the experiment also gathered behavioral data and participant questionnaire scores, which are managed by the `Participant.py` class:
* **TAS-20** (Toronto Alexithymia Scale)
* **PANAS** (Positive and Negative Affect Schedule)
* **AQ-10** (Autism Spectrum Quotient)

### 1. Data Loading (`load_data.py`)
* Loads raw participant data from the Empatica EmbracePlus `.avro` files.
* Parses signals into `pandas` DataFrames, including timestamps and sampling frequencies for:
    * Blood Volume Pulse (BVP)
    * Electrodermal Activity (EDA)
    * Skin Temperature (SKT)
    * 3-axis Accelerometer (ACC)

### 2. Preprocessing (`preprocessing.py`)
The implemented pipeline:
* **BVP:** A 2nd-order Butterworth bandpass filter (1-8 Hz) is applied. Movement artifacts are identified using accelerometer jerk (`|jerk| > 2ms^-3`) and removed, followed by statistical outlier removal (3-sigma rule).
**In practice it was not used**, as the Empatica's algorithms to calculate systolicPeaks lead to better results (the above mentioned pipeline reduces the amount of uncorrupted segments significantly)
* **EDA:** A 1st-order Butterworth lowpass filter (1 Hz) is applied. The signal is then decomposed into its tonic and phasic components using `neurokit2`.
* **SKT:** A 1st-order Butterworth lowpass filter (1 Hz) is applied, followed by statistical outlier removal.

### 3. Segmentation (`segmentation.py`)
* The continuous, preprocessed signals are segmented into windows of a fixed duration.
* A key function, `find_uncorrupted_segments`, ensures that only high-quality segments are used for feature extraction (e.g., segments with no `NaN` values in BVP and sufficient data points in SKT).
**This class was not used for the final pipeline**

### 4. Feature Extraction (`feature_extraction.py` & `helper.py`)
* A wide range of features is extracted from each valid segment using the **NeuroKit2** library and custom functions.
    * **HRV (from IBI):** Time-domain, frequency-domain, and non-linear features (e.g., RMSSD, LF/HF ratio, SampEn).
    * **EDA (from Phasic):** Event-related features, such as SCR (Skin Conductance Response) count, amplitude, and rise time.
    * **SKT:** Statistical features, including mean, standard deviation, and the slope of the signal (linear regression).

### 5. Classification (`classifier.py`)
* A **Random Forest Classifier** is used to predict the target emotion.
* **Recursive Feature Elimination with Cross-Validation (RFECV)** is employed to automatically select the most predictive and relevant features from the large feature set.
* The model's performance is validated using **Leave-One-Out (LOO)** cross-validation, which is a robust method for datasets with a limited number of participants.

