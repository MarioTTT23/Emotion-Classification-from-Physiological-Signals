import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk

from Participant import Participant
from feature_extraction import skt_features

def extract_segments(participant, WD, SP):
    ibi_segments = []
    eda_segments = []
    skt_segments = []

    for index in participant.get_indexes():
        ibi_segments.extend(
            participant.get_ibi_segments(
                index=index,
                WD=WD,
                SP=SP
            )
        )
        eda_segments.extend(
            participant.get_eda_segments(
                index=index,
                WD=WD,
                SP=SP
            )
        )
        skt_segments.extend(
            participant.get_skt_segments(
                index=index,
                WD=WD,
                SP=SP
            )
        )
    return ibi_segments, eda_segments, skt_segments

def extract_features(ibi_segments, eda_segments, skt_segments):
    X = []
    for i in range(len(ibi_segments)):

        sample_eda_features = nk.eda_eventrelated(
            {'0': nk.eda_process(nk.eda_simulate(duration=10), sampling_rate=4)[0]}
        ).iloc[0]
        eda_feature_names = sample_eda_features.index
        eda_empty_features = pd.Series(np.nan, index=eda_feature_names)

        try:
            # Try to extract EDA features as before
            eda_feats = nk.eda_eventrelated(
                {'0': nk.eda_process(eda_segments[i]['value'], sampling_rate=4)[0]}
            ).replace(np.nan, 0).iloc[0]
        except ValueError as e:
            # If a ValueError occurs (likely due to no peaks), log it and use placeholders
            print(f"Warning: Could not process EDA segment {i}. Reason: {e}. Filling with NaNs.")
            eda_feats = eda_empty_features
        
        X.append(
            pd.concat(
                (
                    nk.hrv(ibi_segments[i]).iloc[0],
                    # nk.eda_eventrelated({'0': nk.eda_process(eda_segments[i]['value'], show=True, sampling_rate=4)[0]}).replace(np.nan, 0).iloc[0],
                    eda_feats,
                    skt_features(skt_segments[i], Fs_skt=1)
                )
            )
        )
    return pd.DataFrame(X)
