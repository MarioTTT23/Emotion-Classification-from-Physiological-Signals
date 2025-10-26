import numpy as np
import pandas as pd
import neurokit2 as nk
from io import StringIO
import matplotlib
import scipy
from scipy.signal import butter, lfilter, freqz, sosfiltfilt, sosfilt
from scipy.signal import argrelextrema, welch, find_peaks
from scipy.stats import linregress
import matplotlib.pyplot as plt
from avro.datafile import DataFileReader
from avro.io import DatumReader

########
### SKT
########

def skt_features(skt_segment: pd.DataFrame, Fs_skt: int = 4):
    features = pd.Series()
    skt_df = skt_segment['value'].copy().dropna()
    features['SKTstd'] = np.std(skt_df)
    x = np.linspace(0, len(skt_df), len(skt_df))
    res = linregress(x, skt_df)
    features['SKTslope'] = res.slope
    # features['SKTintercept'] = res.intercept
    return features

########
### BVP
########

def bvp_nk_features(bvp_segment: pd.DataFrame, Fs_bvp: float = 64):
    """
    Returns Neurokit2 direct calculated features (ppg_process() -> ppg_analyze)
    """
    bvp = bvp_segment['value'].copy()
    signal, info = nk.ppg_process(bvp, sampling_rate=Fs_bvp)
    features = nk.ppg_analyze(signal, sampling_rate=Fs_bvp)
    return features.iloc[0]

########
### EDA
########

def eda_clean_features(eda_segment: pd.DataFrame, Fs_eda: float):
    """
    Returns a pd.Series with EDA's basic statistical features and the ones mentioned in 
    Koelstra, S., Muhl, C., Soleymani, M., Lee, J. S., Yazdani, A., Ebrahimi, T., ... & Patras, I. (2011). 
    Deap: A database for emotion analysis; using physiological signals.
    """
    feat = pd.Series()
    eda_df = eda_segment['EDA_Clean'].copy().dropna()
    # Basic statistical features
    feat['EDAmean'] = np.mean(eda_df)
    feat['EDAstd'] = np.std(eda_df)
    feat['EDAmin'] = np.min(eda_df)
    feat['EDAmax'] = np.max(eda_df)
    # DEAP features
    ## First Derivative
    EDAFD = np.gradient(eda_df)
    feat['EDAFDmean'] = np.mean(EDAFD)
    feat['EDAFDNegativesMean'] = np.mean(EDAFD[EDAFD<0])
    feat['EDAFDNegativesRatio'] = len(EDAFD[EDAFD<0])/len(EDAFD)

    minima_indices = argrelextrema(eda_df.values, np.less)[0]
    minima_indices = eda_df.iloc[minima_indices].index
    feat['EDAminimas'] = len(minima_indices)

    maxima_indices = argrelextrema(eda_df.values, np.greater)[0]
    maxima_indices = eda_df.iloc[maxima_indices].index
    rise_durations = []
    # Iterate through minima and find the next occurring maximum
    for min_idx in minima_indices:
        # Find maxima that occur AFTER the current minimum
        subsequent_maxima = maxima_indices[maxima_indices > min_idx]
        if len(subsequent_maxima) > 0:
            # Take the first maximum after the minimum
            max_idx = subsequent_maxima[0]
            # Ensure the value at the maximum is actually greater than the minimum
            if eda_df.loc[max_idx] > eda_df.loc[min_idx]:
                # Because of the sampling rate usually in range [0.25 - 1]
                duration = eda_segment.loc[max_idx]['timestamp'] - eda_segment.loc[min_idx]['timestamp']
                # duration = df.iloc[max_idx]['timestamp'] - df.iloc[min_idx]['timestamp']
                rise_durations.append(duration)
    feat['EDARiseTimeMean'] = np.mean(rise_durations)

    # I didn't found what the 10 stands for:
    # total spectral power within a defined frequency band -> good characterization of stochastic signal
    ## Calculate the Power Spectral Density (PSD)
    frequencies, psd = welch(eda_df, fs=Fs_eda, nperseg = 32, noverlap=16)
    ## Integrate within the desired frequency band
    band_indices = np.where((frequencies >= 0) & (frequencies <= 2.4))
    band_frequencies = frequencies[band_indices]
    band_psd = psd[band_indices]
    if len(band_frequencies) > 1:
        d = band_frequencies[1] - band_frequencies[0] 
        total_power = np.sum(band_psd) * d
    elif len(band_frequencies) == 1:
        # Edge case for single bin
        total_power = band_psd[0] * (frequencies[1] - frequencies[0]) if len(frequencies) > 1 else band_psd[0] 
    else:
        total_power = 0.0
    feat['EDApower'] = total_power

    # zero crossing rate and mean of peaks magnitude of (SCSR [0-0.2]Hz, SCVSR [0-0.8]Hz)  
    # How a [0-0.2]Hz Band Could Cross Zero?
    # The key here is filtering and baseline removal -> Perform on eda_phasic
    # HIGHLY DEPENDENT ON SCR EXTRACTION METHOD -> no convex optimization approach
    sos = butter(N=2, Wn = 0.2, btype='lowpass', fs=Fs_eda, output='sos')
    SCSR = sosfiltfilt(sos, eda_segment['EDA_Phasic'].dropna())
    feat['SCSRzcr'] = len(np.where(np.diff(np.sign(SCSR)))[0])/len(SCSR)
    feat['SCSRmpa'] = calculate_mean_peak_magnitude(SCSR, fs=Fs_eda, distance_threshold_ms=500)

    sos = butter(N=2, Wn = 0.8, btype='lowpass', fs=4, output='sos')
    SCVSR = sosfiltfilt(sos, eda_segment['EDA_Phasic'].dropna())
    feat['SCVSRzcr'] = len(np.where(np.diff(np.sign(SCVSR)))[0])/len(SCVSR)
    feat['SCVSRmpa'] = calculate_mean_peak_magnitude(SCVSR, fs=Fs_eda, distance_threshold_ms=500)

    # Choi 2012
    feat['SCLmean'] = np.mean(eda_segment['EDA_Tonic'])
    feat['SCLlin'] = eda_segment['timestamp'].corr(eda_segment['EDA_Tonic'])
    feat['EDAPhasicstd'] = np.std(eda_segment['EDA_Phasic'])

    return feat

def calculate_mean_peak_magnitude(signal, fs, distance_threshold_ms=50):
    """
    Calculates the mean magnitude of peaks (local maxima and minima).
    Optionally, includes a distance threshold to avoid very closely spaced 'peaks'
    that might be noise.
    """
    # Find positive peaks (local maxima)
    # distance: Minimum number of samples between peaks.
    # Convert ms threshold to samples: distance = (fs * distance_threshold_ms) / 1000
    min_distance_samples = int((fs * distance_threshold_ms) / 1000)
    if min_distance_samples < 1:
        min_distance_samples = 1 # At least 1 sample distance

    peaks_positive_indices, _ = find_peaks(signal, distance=min_distance_samples)

    # Find negative peaks (local minima) by inverting the signal
    peaks_negative_indices, _ = find_peaks(-signal, distance=min_distance_samples)

    # Get magnitudes of all peaks (absolute values)
    all_peak_magnitudes = np.abs(signal[peaks_positive_indices])
    all_peak_magnitudes = np.concatenate((all_peak_magnitudes, np.abs(signal[peaks_negative_indices])))

    if len(all_peak_magnitudes) == 0:
        return 0.0
    return np.mean(all_peak_magnitudes)