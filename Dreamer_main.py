import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, accuracy_score
import matplotlib.pyplot as plt

# Preprocessing and feature extraction for EEG data
def preprocessing_and_feature_extraction_EEG(eeg_data, sampling_rate, frequency_bands, features):
    """
    Preprocess EEG data by applying FIR filtering and extracting power spectral density (PSD) features.

    Parameters:
    - eeg_data (array-like): Raw EEG data.
    - sampling_rate (int): Sampling rate of the EEG data.
    - frequency_bands (list of tuples): Frequency bands for filtering (e.g., [(1, 4), (4, 8)]).
    - features (list): List to store extracted features.
    """
    for band in frequency_bands:
        # Apply FIR filtering for each frequency band
        b = signal.firwin(numtaps=101, cutoff=band, fs=sampling_rate, pass_zero=False)
        filtered_data = signal.lfilter(b, 1.0, eeg_data)
        
        # Calculate Power Spectral Density (PSD) using Welch's method
        freqs, psd = signal.welch(filtered_data, fs=sampling_rate, nperseg=256)
        
        # Append the PSD as a feature
        features.append(np.mean(psd))

# Preprocessing and feature extraction for ECG data
def preprocessing_and_feature_extraction_ECG(ecg_data, sampling_rate):
    """
    Extract ECG features using neurokit2 for processing.

    Parameters:
    - ecg_data (array-like): Raw ECG data.
    - sampling_rate (int): Sampling rate of the ECG data.

    Returns:
    - DataFrame with extracted features.
    """
    processed_ecg, info = nk.ecg_process(ecg_data, sampling_rate=sampling_rate)
    return processed_ecg

# Extract EEG baseline features and save to CSV
def feature_extraction_EEG_end_baseline(eeg_end_baseline, sampling_rate, frequency_bands, output_csv):
    """
    Extract EEG features from the end baseline and save them to a CSV file.

    Parameters:
    - eeg_end_baseline (DataFrame): EEG data for the end baseline.
    - sampling_rate (int): Sampling rate of the EEG data.
    - frequency_bands (list of tuples): Frequency bands for filtering.
    - output_csv (str): Path to save the output CSV file.
    """
    eeg_features = []
    for column in eeg_end_baseline.columns:
        channel_data = eeg_end_baseline[column].to_numpy()
        channel_features = []
        preprocessing_and_feature_extraction_EEG(channel_data, sampling_rate, frequency_bands, channel_features)
        eeg_features.append(channel_features)

    # Create DataFrame for features and save to CSV
    features_df = pd.DataFrame(eeg_features, columns=[f'band_{i}' for i in range(len(frequency_bands))])
    features_df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")

# Extract participant data and process it into a DataFrame
def Participants_Data(participants_info):
    """
    Extract participant metadata such as age and emotional state.

    Parameters:
    - participants_info (DataFrame): DataFrame with participant details.

    Returns:
    - DataFrame with processed participant data.
    """
    participants_data = []
    for index, row in participants_info.iterrows():
        age = row['Age']
        emotion = row['Emotion']
        participants_data.append({'Age': age, 'Emotion': emotion})
    
    return pd.DataFrame(participants_data)

# Example usage
if __name__ == "__main__":
    # Sample EEG and ECG data for demonstration (replace with actual data)
    eeg_sample_data = pd.DataFrame(np.random.randn(1000, 4), columns=['Ch1', 'Ch2', 'Ch3', 'Ch4'])
    ecg_sample_data = np.random.randn(1000)
    
    # Parameters
    sampling_rate = 1000  # Hz
    frequency_bands = [(1, 4), (4, 8), (8, 12), (12, 30)]
    
    # Extract and save EEG features
    feature_extraction_EEG_end_baseline(eeg_sample_data, sampling_rate, frequency_bands, 'eeg_features.csv')
    
    # Extract ECG features
    ecg_features = preprocessing_and_feature_extraction_ECG(ecg_sample_data, sampling_rate)
    print(ecg_features.head())

    # Sample participants info
    participants_info = pd.DataFrame({
        'Age': [25, 30, 22],
        'Emotion': ['Happy', 'Sad', 'Neutral']
    })

    participants_df = Participants_Data(participants_info)
    print(participants_df)

    # Sample model predictions and true labels for evaluation (for demonstration purposes)
    y_true = [1, 0, 1, 1, 0]  # Replace with actual true labels
    y_pred = [1, 0, 1, 0, 0]  # Replace with actual predicted labels

    # Calculate metrics
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    # Print metrics
    print(f"Precision: {precision:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
