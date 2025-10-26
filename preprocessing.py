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

def bandpass_filtering_bvp(bvp_df_orig: pd.DataFrame, 
                           order: int = 2, cutoff_hz = [1,8], sampling_rate: int = 64):
    bvp_df = bvp_df_orig.copy()
    # 1. BVP Filtering (4th-order Butterworth bandpass: 1-8 Hz)
    # The order will get multiplied by 2 because of sosfiltfilt
    sos = butter(order, cutoff_hz, btype='bandpass', fs=sampling_rate, output='sos')
    bvp_df['bvp_filtered'] = sosfiltfilt(sos, bvp_df['value'])
    return bvp_df

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

    # Reason for the scaling: conversionFactor
    # IMU params available in rawData from avro file
    acc_df[['x','y','z']] = acc_df[['x','y','z']] * 0.0004879999905824661

    # Compute jerk
    dt_acc = 1.0 / Fs_acc
    for axis in acc_cols:
        acc_df[f'jerk_{axis}'] = np.gradient(acc_df[axis].to_numpy(), dt_acc)

    # 'movement_epoch' = True if any axis jerk > threshold
    acc_df['movement_epoch'] = False
    for axis in acc_cols:
        acc_df['movement_epoch'] = acc_df['movement_epoch'] | (np.abs(acc_df[f'jerk_{axis}']) > jerk_threshold)

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
           corruption_window_points, jerk_threshold) -> pd.DataFrame:
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

    sos = butter(N=2, Wn=[1,8], btype='bandpass', output='sos', fs=bvp_fs)
    bvp['bvp_filtered'] = sosfiltfilt(sos, bvp['value'])
    
    bvp = movement_based_outlier_removal_bvp(bvp_df_orig=bvp, acc_df_orig=acc, 
                                             corruption_window_points=corruption_window_points, 
                                             jerk_threshold=jerk_threshold,
                                             Fs_bvp=bvp_fs,Fs_acc=acc_fs)
    bvp = statistical_based_outlier_removal_bvp(bvp_df_orig=bvp)
    return bvp