import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
import scikit_posthocs as sp
from itertools import combinations

import neurokit2 as nk

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict, LeaveOneOut
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from Participant import Participant
import preprocessing as pp

# from sklearn.ensemble import RandomForestClassifier

# Method 1: Cleaning using Jerk, SQI and Peak Correction
def manual_cleaning(ppg_df, acc_df, jerk_threshold=2.0, sqi_threshold=0.91, interval_min=0.3):
    ppg_df = ppg_df.copy()
    ppg_df['value'] = ppg_df['value'] * -1
    acc_df = acc_df.copy()
    # Save a copy of the original PPG signal before processing for use in the alternative cleaning method if needed
    ppg_raw = ppg_df.copy()

    # Compute jerk, apply bandpass filter and threshold
    jerk_values = np.abs(pp.compute_jerk(acc_df))
    ppg_df['value'] = pp.bvp_pp(ppg_df, acc_df, bvp_fs= 64.00064849853516, acc_fs= 64.00064849853516,
              corruption_window_points=200, jerk_threshold=jerk_threshold)['bvp_processed']
    
    # Safety check: If more than 80% of the signal was masked, skip
    processed_values = ppg_df['value'].fillna(0).values
    if np.count_nonzero(processed_values) < (0.2 * len(processed_values)):
        print(f"Warning: Signal too noisy. Skipping peaks.")
        return ppg_df, jerk_values, np.zeros(len(ppg_df)), [], []
    
    # Find peaks and onsets
    try:
        peaks_dict = nk.ppg_findpeaks(processed_values, sampling_rate=64, method='charlton', show=False)
        # Check if any peaks were actually found
        if len(peaks_dict['PPG_Peaks']) < 2:
            print("Warning: Too few peaks found")
            return ppg_df, jerk_values, np.zeros(len(ppg_df)), [], []
    except Exception as e:
        print(f"Charlton find peaks failed ({e})")
        _, peaks_dict = neurokit_cleaning(
            ppg_df=ppg_raw, 
            sampling_rate=64
        )

    # Compute SQI and apply mask
    quality = nk.ppg_quality(ppg_df['value'].fillna(0).values, peaks_dict['PPG_Peaks'], sampling_rate=64, method='templatematch')
    low_quality_mask = quality <= sqi_threshold
    ppg_df.loc[low_quality_mask, 'value'] = np.nan

    # Fix double peaks and onsets
    info_peaks, fixed_peaks = nk.signal_fixpeaks(np.array(peaks_dict['PPG_Peaks']), sampling_rate=64, method='neurokit',show=False, interval_min=interval_min)
    print('In seconds - missed peaks:', np.array(info_peaks['missed'])/64, 'extra peaks:', np.array(info_peaks['extra'])/64)
    info_onsets, fixed_onsets = nk.signal_fixpeaks(np.array(peaks_dict['PPG_Onsets']), sampling_rate=64, method='neurokit',show=False, interval_min=interval_min)
    print('In seconds - missed onsets:', np.array(info_onsets['missed'])/64, 'extra onsets:', np.array(info_onsets['extra'])/64)

    return ppg_df, jerk_values, quality, fixed_peaks, fixed_onsets

# Method 2: Simple SQI-based Cleaning (No Jerk or Peak Correction)
def neurokit_cleaning(ppg_df, sampling_rate=64):
    ppg_signal = ppg_df['value'].fillna(0).values *-1

    ppg_cleaned = pp.bandpass_filtering_bvp(ppg_signal, cutoff_hz=[1,16])
    
    # Find peaks
    method_peaks = "charlton"
    peaks_signal, info = nk.ppg_peaks(
        ppg_cleaned, sampling_rate=sampling_rate, method=method_peaks, correct_artifacts=True, show=False
    )
    info["sampling_rate"] = sampling_rate

    # Rate computation
    rate = nk.signal_rate(info["PPG_Peaks"], sampling_rate=sampling_rate, desired_length=len(ppg_cleaned))

    # Assess signal quality
    method_quality = 'templatematch' 
    quality = nk.ppg_quality(
        ppg_cleaned,
        peaks=info["PPG_Peaks"],
        sampling_rate=sampling_rate,
        method=method_quality
    )

    # Prepare output
    signals = pd.DataFrame(
        {
            "PPG_Raw": ppg_signal,
            "PPG_Clean": ppg_cleaned,
            "PPG_Rate": rate,
            "PPG_Quality": quality,
            "PPG_Peaks": peaks_signal["PPG_Peaks"].values,
        }
    )
    return signals, info

# Feature extraction (PPG)
def consecutive_pulse_metrics(signal, complete_fiducial_points, fs=64, window_size=10):
    """
    Calculates the variance of systolic areas VSA, heart rate and T_2 for windows of consecutive pulses.
    Discards windows where pulses are not strictly consecutive (gaps found).
    Returns start and end of each pulse sequence as well
    """
    pulses = []
    areas = pp.systolic_area(signal, complete_fiducial_points)
    

    # Sliding window to find blocks of 10 consecutive pulses
    i = 0
    while i <= len(areas) - window_size:
        window = areas[i : i + window_size]
        
        # Check consecutiveness: 
        is_consecutive = True
        for j in range(window_size - 1):
            if window[j]['next_onset'] != window[j+1]['onset']:
                is_consecutive = False
                break
        
        if is_consecutive:
            # VSA (Variance of Systolic Areas)
            area_values = [pulse['area'] for pulse in window]
            block_variance = np.var(area_values)
            # HR (Heart Rate)
            peaks = [pulse['peak'] for pulse in window]
            # heart_rate = np.var(np.diff(peaks))
            heart_rate = np.mean(np.diff(peaks))
            # T_2
            next_onsets = [pulse['next_onset'] for pulse in window]
            t_2s = (np.array(next_onsets) - np.array(peaks)) / fs 
            t_2_mean = np.var(t_2s)

            pulses.append({
                'VSA': block_variance,
                'HR': heart_rate,
                'T_2': t_2_mean,
                'start': window[0]['onset'],
                'end': window[window_size-1]['next_onset']
            })
            i += window_size
        else:
            i += 1
    return pulses

# Features
features = ["HR", "VSA"]
# SVM kernel
kernel = 'linear'

for method in [1, 2]:
    for scaling in ['Single', 'Blockwise']:
        for min_dataset_size in [20, 30, 40]:
            for C in [0.01, 0.1, 1.0]:
                file_name = f'linear_2d_{method}_{scaling}_{min_dataset_size}_{C}.csv'

                # Segmentation params
                delta_start = -1
                extra_duration = 2

                to_store = []
                for pid in [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]:
                    p = Participant(participantID=pid, rawData=True)
                    print(f'Currently processing participant {pid}')

                    for condition in ['audio-only', 'image-only', 'audio+image']:
                        for te1 in ['Happiness', 'Anger']:
                            for te2 in ['Anger', 'Sadness']:
                                # Skip Anger vs Anger
                                if te1 == te2:
                                    continue
                                # Segmentation
                                segments = {} 
                                segments[te1] = {}
                                segments[te2] = {}
                                if condition == 'audio-only':
                                    condition_markers = p.markers[p.markers['condition'] == 'audio-only']
                                elif condition == 'image-only':
                                    condition_markers = p.markers[p.markers['condition'] == 'image-only']
                                else:
                                    condition_markers = p.markers[p.markers['condition'] == 'audio+image']
                                te1_markers = condition_markers[condition_markers['target_emotion'] == te1]
                                te2_markers = condition_markers[condition_markers['target_emotion'] == te2]
                                if condition == 'audio-only':
                                    te1_onsets = te1_markers['audio_onset'].values / 1000
                                    te2_onsets = te2_markers['audio_onset'].values / 1000
                                    # Audio-only lasts for 2 minutes, get only first and third audio onset for each emotion
                                    segments[te1]['b1'] = p.get_rawData_from_timestamp(te1_onsets[0], delta_start=delta_start, duration=120+extra_duration)
                                    segments[te1]['b2'] = p.get_rawData_from_timestamp(te1_onsets[2], delta_start=delta_start, duration=120+extra_duration)
                                    segments[te2]['b1'] = p.get_rawData_from_timestamp(te2_onsets[0], delta_start=delta_start, duration=120+extra_duration)
                                    segments[te2]['b2'] = p.get_rawData_from_timestamp(te2_onsets[2], delta_start=delta_start, duration=120+extra_duration)
                                else:
                                    b_i=1
                                    for index in te1_markers.index:
                                        start = p.get_timestamp_from_index(index, img_idx=0)
                                        segments[te1][f'b{b_i}'] = p.get_rawData_from_timestamp(start, delta_start=delta_start, duration=60+extra_duration)
                                        b_i += 1
                                    b_i=1
                                    for index in te2_markers.index:
                                        start = p.get_timestamp_from_index(index, img_idx=0)
                                        segments[te2][f'b{b_i}'] = p.get_rawData_from_timestamp(start, delta_start=delta_start, duration=60+extra_duration)
                                        b_i += 1
                                if condition == 'audio-only':
                                    configs = [
                                        {'te': te1, 'block': 'b1'},
                                        {'te': te2, 'block': 'b1'},
                                        {'te': te1, 'block': 'b2'},
                                        {'te': te2, 'block': 'b2'}
                                    ]
                                else:
                                    configs = [
                                        {'te': te1, 'block': 'b1'},
                                        {'te': te2, 'block': 'b1'},
                                        {'te': te1, 'block': 'b2'},
                                        {'te': te2, 'block': 'b2'},
                                        {'te': te1, 'block': 'b3'},
                                        {'te': te2, 'block': 'b3'}
                                    ]

                                # Processing / Feature Extraction
                                # Iterate over each tuple (target emotion, block) and extract features
                                all_emotion_data = []
                                for i, config in enumerate(configs, start=1):
                                    te, block = config['te'], config['block']
                                    ppg_raw = segments[te][block]['bvp'].copy()
                                    eda_raw=segments[te][block]['eda'].copy()
                                    eda_phasic=nk.eda_phasic(eda_raw['value'], sampling_rate=4, method='cvxEDA')['EDA_Phasic']
                                    domain = ppg_raw['timestamp'] - ppg_raw['timestamp'].values[0]
                                    if method == 1:
                                        acc_df=segments[te][block]['acc'].copy()
                                        ppg_df, jerk_values, quality, fixed_peaks, fixed_onsets = manual_cleaning(
                                            ppg_df=ppg_raw, 
                                            acc_df=acc_df,
                                            jerk_threshold=2.0, sqi_threshold=0.95, interval_min=0.3
                                            )
                                        if len(fixed_onsets) == 0:
                                            print(f"Skipping {te} {block}: No onsets detected (signal too noisy).")
                                            continue
                                        mask = ppg_df['value'].notna()
                                        dic_notches, fiducial_points = pp.detect_dicrotic_notch(
                                            np.array(ppg_df['value'].fillna(0)), fixed_peaks, fixed_onsets, fs=64, 
                                            window_length=6, beta = 0.1
                                            )
                                        ppg_df['value'] = pp.remove_baseline_onsets(ppg_df['value'].fillna(0), np.array(fixed_onsets), fs = 64)[0]
                                        ppg_df.loc[~mask, 'value'] = np.nan
                                        fiducial_points = pp.filter_incomplete_pulses(ppg_df['value'], fiducial_points)

                                        ppg_signal = ppg_df['value'].values
                                        # domain = ppg_df['timestamp']-ppg_df['timestamp'].values[0]
                                    elif method == 2:
                                        signals, info = neurokit_cleaning(
                                            ppg_df=ppg_raw, 
                                            sampling_rate=64
                                        )
                                        ppg_signal = signals['PPG_Clean'].values
                                        quality = signals['PPG_Quality'].values
                                        fixed_peaks = info['PPG_Peaks']
                                        fixed_onsets = info['PPG_Onsets']

                                        dic_notches, fiducial_points = pp.detect_dicrotic_notch(
                                            ppg_signal, fixed_peaks, fixed_onsets, fs=64, 
                                            window_length=6, beta=0.1
                                            )
                                        ppg_signal = pp.remove_baseline_onsets(ppg_signal, np.array(fixed_onsets), fs = 64)[0]
                                        fiducial_points = pp.filter_incomplete_pulses(ppg_signal, fiducial_points)
                                    window_metrics = consecutive_pulse_metrics(ppg_signal, fiducial_points, fs=64)
                                    for v in window_metrics:
                                        all_emotion_data.append({
                                            # 'Emotion': f"{te}_{block}",
                                            'Emotion': te,
                                            'Block': block,
                                            'VSA': v['VSA'],
                                            'HR': v['HR'],
                                            'T_2': v['T_2'],
                                            'ComEDA': nk.entropy_sample(
                                                eda_phasic.values[int((v['start']/64)*4):int((v['end']/64)*4)], 
                                                delay=1, dimension=1, tolenrance='sd'
                                                )[0]
                                        })
                                df = pd.DataFrame(all_emotion_data)
                                print(f"dataset size: {len(df)} rows.")
                                # Outlier Removal for method 2
                                if method == 2:
                                    for feature in features:
                                        Q1 = df[feature].quantile(0.25)
                                        Q3 = df[feature].quantile(0.75)
                                        IQR = Q3 - Q1
                                        
                                        # Define bounds
                                        lower_bound = Q1 - 2 * IQR
                                        upper_bound = Q3 + 2 * IQR
                                        
                                        # Filter the dataframe
                                        df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
                                    
                                    print(f"Outliers removed. New dataset size: {len(df)} rows.")
                            
                                X = df[features].values
                                y = df['Emotion'].values
                                groups = df['Block'].values 

                                # --- Safety Checks ---
                                if len(df) < min_dataset_size:
                                    print(f"Skipping {pid} in {condition}: Only {len(df)} rows. Need at least 20.")
                                    continue

                                unique_classes = np.unique(y)
                                if len(unique_classes) < 2:
                                    print(f"Skipping {pid} in {condition}: Only found one class ({unique_classes}).")
                                    continue

                                if len(np.unique(groups)) < 2:
                                    print(f"Skipping {pid} in {condition}: Need at least 2 blocks for LOGO-CV.")
                                    continue

                                # Scaling
                                if scaling == 'Single':
                                    clf = make_pipeline(
                                        StandardScaler(), 
                                        SVC(
                                            kernel=kernel, 
                                            C=C, gamma='auto', class_weight='balanced')
                                    )
                                else: # Blockwise Scaling
                                    scaler = StandardScaler()
                                    X_scaled = np.zeros_like(X)
                                    for block_id in np.unique(groups):
                                        mask = groups == block_id
                                        X_scaled[mask] = scaler.fit_transform(X[mask])
                                    X = X_scaled

                                    clf = make_pipeline(
                                        SVC(
                                            kernel=kernel, 
                                            C=C, gamma='auto', class_weight='balanced')
                                    )

                                # Perform Block-CV
                                logo = LeaveOneGroupOut()
                                try:
                                    y_pred = cross_val_predict(clf, X, y, groups=groups, cv=logo)
                                    block_acc = accuracy_score(y, y_pred)
                                    block_bal_acc = balanced_accuracy_score(y, y_pred)
                                    print(f'Participant {pid} in {condition} for Block-CV ({te1} vs {te2}): {block_acc} (Balanced: {block_bal_acc})')
                                except ValueError as e:
                                    print(f"Skipping P{pid} due to internal CV error: {e}")
                                    continue

                                # Perform LOO-CV
                                loo = LeaveOneOut()
                                y_pred = cross_val_predict(clf, X, y, cv=loo)
                                loo_acc = accuracy_score(y, y_pred)
                                loo_bal_acc = balanced_accuracy_score(y, y_pred)
                                print(f'Participant {pid} in {condition} for LOO-CV ({te1} vs {te2}): {loo_acc} (Balanced: {loo_bal_acc})')

                                to_store.append({
                                    'id': pid,
                                    'condition': condition,
                                    'te1': te1,
                                    'te2': te2,
                                    'block_acc': block_acc,
                                    'block_bal_acc': block_bal_acc,
                                    'loo_acc': loo_acc,
                                    'loo_bal_acc': loo_bal_acc
                                })
                to_store_df = pd.DataFrame(to_store)
                to_store_df.to_csv(file_name, index=False)




