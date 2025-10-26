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

def segment(startTS: float, endTS: float, df: pd.DataFrame):
    """
    Both the start and end timestamps are Unix timestamps in ms. 
    The input dataframe must contain one 'timestamp' column
    """
    if isinstance(df, pd.Series):
        condition = (df >= startTS) &\
                (df <= endTS)
        return df[condition].reset_index(drop=True)
    condition = (df['timestamp'] >= startTS) &\
                (df['timestamp'] <= endTS)
    return df[condition].reset_index(drop=True)

def multiple_segments(startTS: float, endTS: float, segment_duration: float, between_segments_distance: float, 
                      signals: dict, Fs_bvp: float, Fs_skt: float, cut_first_x_sec: int=0):
    """
    Input: start and end timestamps (TS) of the time period to be segmented, duration, 
        and the separation(+)/overlap(-) between segments IN SECONDS. 
        signals contains at least [bvp_pp, skt_pp].
    Output: timestamps corresponding to the start and end-points of the segments
    """
    bvp_pp = signals['bvp_pp'].copy()
    skt_pp = signals['skt_pp'].copy()

    if segment_duration <= -between_segments_distance:
        return 'invalid segment_duration or between_segments_distance'

    if cut_first_x_sec>0:
        startTS = startTS + cut_first_x_sec
    bvp_epoch = bvp_pp[(bvp_pp['timestamp'] >= startTS) &\
                         (bvp_pp['timestamp'] <= endTS)]
    skt_epoch = skt_pp[(skt_pp['timestamp'] >= startTS) &\
                         (skt_pp['timestamp'] <= endTS)]
    
    multiple_segments = []
    uncorrupted_segments_indices = find_uncorrupted_segments(bvp_pp=bvp_epoch, skt_pp=skt_epoch, 
                                                             Fs_bvp=Fs_bvp, Fs_skt=Fs_skt, 
                                                            segment_duration=segment_duration)
    
    # filter segments with non-desirable overlap/between segments distance
    if len(uncorrupted_segments_indices) == 0:
        return f'no uncorrupted segments found in epoch with startTS {startTS}'
    if len(uncorrupted_segments_indices) == 1:
        return (bvp_epoch['timestamp'][uncorrupted_segments_indices[0][0]], bvp_epoch['timestamp'][uncorrupted_segments_indices[0][1]])
    
    multiple_segments.append((bvp_epoch['timestamp'][uncorrupted_segments_indices[0][0]], 
                              bvp_epoch['timestamp'][uncorrupted_segments_indices[0][1]]))
    for isegment in uncorrupted_segments_indices[1:]:
        # if startTS of isegment >= endTS of last element in multiple segments + between seg dist
        if bvp_epoch['timestamp'][isegment[0]] >= (multiple_segments[len(multiple_segments)-1][1] + between_segments_distance):
            multiple_segments.append(
                (bvp_epoch['timestamp'][isegment[0]], 
                 bvp_epoch['timestamp'][isegment[1]])
            )
    return multiple_segments

def middle_segment(startTS: float, endTS: float, segment_duration: float, signals: dict, Fs_bvp: float, Fs_skt: float, cut_first_30sec:bool=True):
    """
    Input: start and end timestamps (TS) of the time period to be segmented, and duration, IN SECONDS. 
        signals contains at least [bvp_pp, skt_pp].
    Output: One element list, containing a tuple with the timestamps corresponding to the start and end of the segment
    Following the method mentioned in https://doi.org/10.3390/bioengineering10111308, 
    1) cut 30 first seconds (avoidable with cut_first_30sec)
    2) identify uncorrupted segments with the required length
    3) select the segment closest to the midpoint
    """
    bvp_pp = signals['bvp_pp'].copy()
    skt_pp = signals['skt_pp'].copy()


    if cut_first_30sec:
        startTS = startTS + 30
    bvp_epoch = bvp_pp[(bvp_pp['timestamp'] >= startTS) &\
                         (bvp_pp['timestamp'] <= endTS)]
    skt_epoch = skt_pp[(skt_pp['timestamp'] >= startTS) &\
                         (skt_pp['timestamp'] <= endTS)]
    uncorrupted_segments_indexes = find_uncorrupted_segments(bvp_pp=bvp_epoch, skt_pp=skt_epoch, 
                                                             Fs_bvp=Fs_bvp, Fs_skt=Fs_skt, 
                                                            segment_duration=segment_duration)

    if bvp_epoch.empty:
        return f'no valid segment for startTS {startTS} and endTS {endTS}'

    # find middle segment (or closest after it)
    pivot = ((bvp_epoch['timestamp'].iloc[-1] + bvp_epoch['timestamp'].iloc[0]) / 2) - segment_duration
    for isegment in uncorrupted_segments_indexes:
        if bvp_epoch['timestamp'][isegment[0]] >= pivot:
            return [(bvp_epoch['timestamp'][isegment[0]],
                    bvp_epoch['timestamp'][isegment[1]])]

    for isegment in reversed(uncorrupted_segments_indexes):
        if bvp_epoch['timestamp'][isegment[0]] <= pivot:
            return [(bvp_epoch['timestamp'][isegment[0]],
                    bvp_epoch['timestamp'][isegment[1]])]

    return f'no match for middle segment with startTS: {startTS}'
    # return segment

def find_uncorrupted_segments(bvp_pp: pd.DataFrame, skt_pp: pd.DataFrame, Fs_bvp: float, Fs_skt: float, segment_duration: float):
    """
    A segment is considered uncorrupted when satifies:
    Condition 1: All BVP values are non-NaN
    Condition 2: At least int(segment_duration * Fs_skt)-1 SKT values
    """
    segment_length = segment_duration*Fs_bvp
    uncorrupted_segments = []
    i=0
    while i + segment_length <= len(bvp_pp):
        start_index = bvp_pp.index[i]
        end_index = bvp_pp.index[i + segment_length - 1]
        bvp_segment = bvp_pp.loc[start_index:end_index]
        skt_segment = skt_pp[(skt_pp['timestamp'] >= bvp_segment['timestamp'].iloc[0]) &\
                             (skt_pp['timestamp'] <= bvp_segment['timestamp'].iloc[-1])]

        # Condition 1: All BVP values are non-NaN
        bvp_not_nan = bvp_segment["bvp_processed"].notna().all()

        # Condition 2: At least int(segment_duration * Fs_skt)-1 SKT values
        skt_not_nan_count = skt_segment['skt_filtered'].notna().sum()
        skt_sufficient = skt_not_nan_count >= (int(segment_duration * Fs_skt) - 1)

        if bvp_not_nan and skt_sufficient:
            uncorrupted_segments.append((start_index, end_index))
        i += 1 
    return uncorrupted_segments