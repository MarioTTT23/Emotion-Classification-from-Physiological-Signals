import numpy as np
import pandas as pd
import neurokit2 as nk
from io import StringIO
import matplotlib
import scipy
from scipy.signal import butter, lfilter, freqz, resample, sosfiltfilt, sosfilt
from scipy.signal import argrelextrema, welch, find_peaks, savgol_filter
from scipy.interpolate import CubicSpline, interp1d
from scipy.stats import linregress
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from avro.datafile import DataFileReader
from avro.io import DatumReader


#################
### Any
#################

def normalize(signal):
    y_min, y_max = np.min(signal), np.max(signal)
    y_norm = (signal - y_min) / (y_max - y_min)
    return y_norm

#################
### PPG pipe
#################

def get_envelopes(signal):
    """Calculates upper and lower envelopes based on first derivative extrema."""
    # Locations of local extrema
    d1 = np.diff(signal)
    max_idx, _ = find_peaks(d1)
    min_idx, _ = find_peaks(-d1)
    
    # Cubic Spline Interpolation
    t = np.arange(len(signal))
    
    # Boundary handling: ensure we have endpoints for the spline
    upper_idx = np.unique(np.clip(np.concatenate(([0], max_idx, [len(signal)-1])), 0, len(signal)-1))
    lower_idx = np.unique(np.clip(np.concatenate(([0], min_idx, [len(signal)-1])), 0, len(signal)-1))
    
    upper_env = CubicSpline(upper_idx, signal[upper_idx])(t)
    lower_env = CubicSpline(lower_idx, signal[lower_idx])(t)
    
    return upper_env, lower_env

def iem_decomposition(y, beta=0.1, window_length=6, max_iter=10):
    """
    Decomposes signal into Non-Stationary (NSTS) and Stationary (STS) parts.
    p=4, n=25 as per Pal et al. 2024
    """
    y_input = y.copy()
    sts = np.zeros_like(y)
    r_prev = np.zeros_like(y)
    
    for i in range(max_iter):
        # Step 1: Smoothing (Savitzky-Golay)
        y_smoothed = savgol_filter(y_input, window_length=window_length, polyorder=4)
        
        # Step 2 & 3: Mean Envelope
        up, lw = get_envelopes(y_smoothed)
        m_i = (up + lw) / 2
        
        # Step 4: Subtract mean to get non-stationary estimate
        r_i = y_input - m_i
        
        # Step 5: Stopping Criterion
        stc = np.abs(np.mean(r_prev**2) - np.mean(r_i**2))
        
        sts += m_i
        y_input = r_i
        r_prev = r_i
        
        if stc < beta:
            break
    print(f'iem iterated {i} times')
    nsts = r_i
    return nsts, sts

def detect_dicrotic_notch(signal, systolic_peaks, onsets, fs=64, beta=0.1, window_length=6):
    """
    Assumes high frequency noise has been filtered
    Returns the indices of dicrotic notches using the IEM-based algorithm. 
    Pal, R., Rudas, A., Kim, S., Chiang, J. N., Barney, A., & Cannesson, M. (2024)
    """
    # Normalize
    y_norm = normalize(signal)
    
    # non-stationary and stationary components
    nsts, sts = iem_decomposition(y_norm, beta=beta, window_length=window_length, max_iter=10)
    
    dn_locations = []
    ### added functionality marked with ###
    fiducial_points = []
    next_onset = None
    
    for peak in systolic_peaks:
        # Condition 1: Valley must be at least 0.1s after systolic peak
        search_start = int(peak + int(0.1 * fs))

        ### store previous onset
        prev_onset = next_onset

        # Search until:
        # a bit before of the next onset (avoids misidentification of an onset as a dn)
        # or until the duration of a typical heartbeat
        next_onset = next((x for x in onsets if x > peak), None) 
        if next_onset is None:
            next_onset = len(nsts)
        end_beat = peak + int(0.6 * fs) 
        search_end = int(np.min([next_onset - (0.1 * fs), end_beat]))
        
        segment = nsts[search_start:search_end]
        valleys, _ = find_peaks(-segment)
        
        for v in valleys:
            actual_idx = search_start + v
            # Condition 2: NSTS value must be less than zero
            if nsts[actual_idx] < 0:
                dn_locations.append(actual_idx)
                ###
                fiducial_points.append((prev_onset, peak, actual_idx, next_onset))
                
                break # Take the first valley that satisfies conditions
    return dn_locations, fiducial_points

def remove_baseline_onsets(signal, onsets, fs=64):
    """
    Removes baseline modulation by interpolating between pulse onsets.
    """
    sig = np.array(signal).flatten()
    t = np.arange(len(sig))
    
    # Include the start and end of the signal for interpolation
    if onsets[0] != 0:
        onsets = np.insert(onsets, 0, 0)
    if onsets[-1] != len(sig) - 1:
        onsets = np.append(onsets, len(sig) - 1)
    
    # Construct the baseline model using Linear Interpolation
    baseline_f = interp1d(onsets, sig[onsets], kind='linear', fill_value="extrapolate")
    baseline_model = baseline_f(t)
    
    # Subtract the baseline
    detrended_sig = sig - baseline_model

    # Pseudo Min-Max norm, ensure onsets are on 0
    min = 0
    norm_sig = (detrended_sig - min) / (np.max(detrended_sig) - np.min(detrended_sig))
    
    return norm_sig, baseline_model

def filter_incomplete_pulses(signal, fiducial_points, fs=64):
    """
    Uses the mask to filter pulses! Do not use fillna on signal!
    """
    filtered_fps = []
    for cuadriple in fiducial_points:
        on, peak, dn, next_on = cuadriple

        # Condition 1: onsets are not None
        if on == None or next_on  == None:
            continue

        # Condition 2: Correctly ordered in time
        if not (on < peak and peak < dn and dn < next_on):
            continue

        # Condition 3: the segment is clean
        segment = signal[int(on) : int(next_on) + 1]
        if np.isnan(segment).any():
            continue

        filtered_fps.append(cuadriple)
    return filtered_fps

def systolic_area(signal, fiducial_points, fs=64):
    """
    Turns a list of fiducial points (onset, peak, dic_notch, next_onset) 
    into a list of (area, onset, peak, next_onset)
    """
    dx = 1.0 / fs
    areas = []

    for cuadriple in fiducial_points:
        on, peak, dn, next_on = cuadriple

        # Integration between onset (on) and dicrotic notch (dn)
        segment = signal[int(on) : int(dn) + 1]
        
        sys_area = simpson(segment, dx=dx)
        # sys_area = np.trapz(segment, dx=dx)
        # print(sys_area, on/64, next_on)

        areas.append({
            'area': sys_area,
            'onset': on,
            'peak': peak,
            'next_onset': next_on
        })
    return areas

#################
### SKT_pp
#################

def skt_pp(skt_df_orig: pd.DataFrame, Fs_skt: int = 4, order: int = 1, cutoff: float = 1.0,
                skt_col: str = 'value', timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """
    Filtering using a 2nd-order Butterworth lowpass filter with a cutoff frequency of 1 Hz.
    The resolution of the SKT signal and the accuracy of the sensor are 0.02 °C (as well). 
    +: WESAD recordings contain corrupted values. Outliers removed based on statistical variance.
    """
    skt_df = skt_df_orig.copy()

    sos = butter(order, cutoff, btype='lowpass', fs=Fs_skt, output='sos')
    skt_df['skt_filtered'] = sosfiltfilt(sos, skt_df['value'])

    skt_df = statistical_based_outlier_removal_skt(skt_df)

    return skt_df

def statistical_based_outlier_removal_skt(skt_df_orig: pd.DataFrame) -> pd.DataFrame:
    # Outlier Removal (Statistical: outside mean +/- 3*std)
    skt_df = skt_df_orig.copy()
    signal_for_stats = skt_df['skt_filtered'].dropna()
    if not signal_for_stats.empty:
        mean_skt = signal_for_stats.mean()
        std_skt = signal_for_stats.std()
        if std_skt > 1e-6 :
            statistical_outlier_high_thresh = mean_skt + (3 * std_skt)
            statistical_outlier_low_thresh = mean_skt - (3 * std_skt)

            outlier_mask_stat = (skt_df['skt_filtered'] < statistical_outlier_low_thresh) | \
                                (skt_df['skt_filtered'] > statistical_outlier_high_thresh)
            print(f"Removed statistical outliers: {outlier_mask_stat.sum()}")
            skt_df.loc[outlier_mask_stat, 'skt_filtered'] = np.nan

    return skt_df

#################
### EDA_pp
#################

def eda_pp2():
    """
    Low pass filtered with a 1000th-order FIR filter with a corner frequency of 1Hz
    https://doi.org/10.1016/j.bspc.2019.101646
    """
    return -1

def eda_pp_neurokit2(eda_df_orig: pd.DataFrame, Fs_eda: int=4):
    """
    Input: EDA DataFrame containing cols ['timestamp','value'], threshold: a float indicating 
    the minimal amplitude for a SCR to be identified as one.
    Output: eda_df (values and filtered signal), eda_signals (nk computed phasic, tonic, and feature indicators), 
    eda_info (dict with info about features), and the cleaned eda signal splitted in phasic and tonic components
    
    Filtering with a 2nd-order Butterworth lowpass filter with a cutoff frequency of 1 Hz.
    Phasic and Tonic separation thorugh NeuroKit METHOD VARIABLE
    SCR extraction using NeuroKit METHOD VARIABLE
    """
    eda_df = eda_df_orig.copy()

    def lowpass_filter_eda(eda_df_orig: pd.DataFrame, Fs_eda: int = 4, 
                       order: int = 1, cutoff: float = 1) -> pd.DataFrame:
        eda_df = eda_df_orig.copy()

        sos = butter(order, cutoff, btype='lowpass', fs=Fs_eda, output='sos')
        eda_df['EDA_Clean'] = sosfiltfilt(sos, eda_df['value'])

        return eda_df
    
    eda_df = lowpass_filter_eda(eda_df, Fs_eda=4, order=1, cutoff=1)
    # split into Tonic and Phasic components
    split_df = nk.eda_phasic(eda_df['EDA_Clean'],sampling_rate=Fs_eda, method="smoothmedian")

    return pd.concat([
        eda_df.reset_index(drop=True),
        split_df.reset_index(drop=True)
        ], axis=1)
    

def eda_pp(eda_df_orig: pd.DataFrame, file_name: str):
    """
    Input: eda contains cols ['timestamp','value']
    This function creates a file containing the raw signal as input for Ledalab.
    Convert the ouput into .txt and export as txt type 1.
    2nd-order Butterworth lowpass filter with a cutoff frequency of 1 Hz.
    (For CDA as Ledalab is needed)
    IMPORTANT: LEDALAB WARNING FOR NON-INTEGER SAMPLING_RATES
    MODIFY!!! ROUND TO 64?
    https://doi.org/10.3390/bioengineering10111308
    """
    eda_df = eda_df_orig.copy()
    eda_df['Marker'] = 0
    eda_df = eda_df[['timestamp', 'value', 'Marker']]
    eda_df.to_csv(file_name, sep=' ', header=False, index=False, float_format='%.6f')
    return file_name


#################
### BVP_pp
#################
def bvp_pp2(bvp_df: pd.DataFrame, acc_df: pd.DataFrame) -> pd.DataFrame:
    """
    100th-order band pass FIR filter with corner frequencies equal to 0.1Hz and 10Hz
    https://doi.org/10.1016/j.bspc.2019.101646
    """

    return -1

def bandpass_filtering_bvp(ppg_signal, 
                           order: int = 2, cutoff_hz = [1,8], sampling_rate: int = 64):
    bvp_df = ppg_signal
    # Filtering (4th-order Butterworth bandpass: 1-8 Hz)
    # The order will get multiplied by 2 because of sosfiltfilt
    sos = butter(order, cutoff_hz, btype='bandpass', fs=sampling_rate, output='sos')
    return sosfiltfilt(sos, ppg_signal)

def compute_jerk(acc_df_orig: pd.DataFrame, Fs_acc: int = 64.00064849853516) -> np.ndarray:
    acc_df = acc_df_orig.copy()
    # Reason for the scaling: conversionFactor
    # IMU params available in rawData from avro file
    acc_df[['x','y','z']] = acc_df[['x','y','z']] * 0.0004879999905824661
    total_acc = np.sqrt(acc_df['x']**2 + acc_df['y']**2 + acc_df['z']**2)
    dt_acc = 1.0 / Fs_acc
    return np.gradient(total_acc, dt_acc)

def movement_based_outlier_removal_bvp(bvp_df_orig: pd.DataFrame, acc_df_orig: pd.DataFrame, 
                                       corruption_window_points: int = 200, jerk_threshold: float = 2.0,
                                       Fs_bvp: int = 64.00064849853516, Fs_acc: int = 64.00064849853516, 
                                       bvp_col: str = 'value', acc_cols: list = ['x', 'y', 'z'],
                                       timestamp_col: str = 'timestamp') -> pd.DataFrame:
    bvp_df = bvp_df_orig.copy()
    acc_df = acc_df_orig.copy()
    g=9.80665

    # If no filter, use value column
    if 'bvp_filtered' in bvp_df:
        bvp_df['bvp_processed'] = bvp_df['bvp_filtered'].copy()
    else:
        print('No filtered bvp signal passed into movement_based_outlier_removal_bvp')
        bvp_df['bvp_processed'] = bvp_df['value'].copy()

    abs_jerk_values = np.abs(compute_jerk(acc_df_orig=acc_df, Fs_acc=Fs_acc))
    acc_df['movement_epoch'] = False
    acc_df['movement_epoch'] = acc_df['movement_epoch'] | (abs_jerk_values > jerk_threshold)

    movement_timestamps_acc = acc_df[acc_df['movement_epoch'] == True][timestamp_col].to_numpy()
    bvp_timestamps_np = bvp_df[timestamp_col].to_numpy()

    # Find indices to corrupt
    indices_to_corrupt_bvp = []
    for mov_ts in movement_timestamps_acc:
        # Find BVP index at or just after the ACC movement timestamp
        start_bvp_idx = np.searchsorted(bvp_timestamps_np, mov_ts, side='left')
        if start_bvp_idx < len(bvp_df):
            # Define the window of BVP points to mark as corrupted
            end_bvp_idx_exclusive = min(len(bvp_df), start_bvp_idx + corruption_window_points)
            indices_to_corrupt_bvp.extend(range(start_bvp_idx, end_bvp_idx_exclusive))

    if indices_to_corrupt_bvp:
        unique_indices_to_corrupt = sorted(list(set(indices_to_corrupt_bvp)))
        
        # Use .iloc for positional indexing if DataFrame index is not standard RangeIndex
        bvp_df.iloc[unique_indices_to_corrupt, bvp_df.columns.get_loc('bvp_processed')] = np.nan

    print(f"Removed outliers due to movement: {len(bvp_df) - bvp_df['bvp_processed'].notna().sum()}")

    return bvp_df

def statistical_based_outlier_removal_bvp(bvp_df_orig: pd.DataFrame) -> pd.DataFrame:
    # Outlier Removal (Statistical: outside mean +/- 3*std)
    bvp_df = bvp_df_orig.copy()
    # Applied to the BVP data that hasn't been marked as NaN due to movement
    signal_for_stats = bvp_df['bvp_processed'].dropna()
    if not signal_for_stats.empty:
        mean_bvp = signal_for_stats.mean()
        std_bvp = signal_for_stats.std()
        # Ensure std_bvp is not zero to avoid division by zero or other issues
        if std_bvp > 1e-6 : # Check if std_bvp is not too small
            statistical_outlier_high_thresh = mean_bvp + (3 * std_bvp)
            statistical_outlier_low_thresh = mean_bvp - (3 * std_bvp)

            outlier_mask_stat = (bvp_df['bvp_processed'] < statistical_outlier_low_thresh) | \
                                (bvp_df['bvp_processed'] > statistical_outlier_high_thresh)
            print(f"Removed statistical outliers: {outlier_mask_stat.sum()}")
            bvp_df.loc[outlier_mask_stat, 'bvp_processed'] = np.nan

    return bvp_df

def bvp_pp(bvp_df: pd.DataFrame, acc_df: pd.DataFrame, 
           bvp_fs: float, acc_fs: float,
           corruption_window_points, jerk_threshold, lowpass = False, cutoff = 15) -> pd.DataFrame:
    """
    Input: bvp contains cols ['timestamp','value']; acc contains cols ['timestamp','x','y','z']
    4th-order Butterworth bandpass filter with cutoff frequencies of 1Hz and 8 Hz 
    to minimize both high-frequency measurement noise and low-frequency movement noise. 
    Movement related outlier elimination (BVP values within a 200-point window (64Hz) after |ACC jerk| > 2ms^-3 ).
    Statistical outliar elimination (data points outside of a threshold of 3*std(BVP))
    https://doi.org/10.3390/bioengineering10111308
    """
    bvp = bvp_df.copy()
    acc = acc_df.copy()

    if lowpass:
        num_samples = len(bvp['value']) * 2
        upsampled_data = resample(bvp['value'], num_samples)
        # as sosfiltfilt is used the order of the filter is doubled
        sos = butter(N=2, Wn=cutoff, btype='lowpass', output='sos', fs=bvp_fs*2)
        filtered_upsampled = sosfiltfilt(sos, upsampled_data)
        bvp['bvp_filtered'] = filtered_upsampled[::2]
    else:
        sos = butter(N=2, Wn=[1,8], btype='bandpass', output='sos', fs=bvp_fs)
        bvp['bvp_filtered'] = sosfiltfilt(sos, bvp['value'])
    
    bvp = movement_based_outlier_removal_bvp(bvp_df_orig=bvp, acc_df_orig=acc, 
                                             corruption_window_points=corruption_window_points, 
                                             jerk_threshold=jerk_threshold,
                                             Fs_bvp=bvp_fs,Fs_acc=acc_fs)
    # bvp = statistical_based_outlier_removal_bvp(bvp_df_orig=bvp)
    return bvp