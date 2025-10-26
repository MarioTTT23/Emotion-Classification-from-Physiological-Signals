import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk
import json

from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFECV, RFE

from Participant import Participant
from feature_extraction import skt_features
from helper import extract_segments, extract_features

import pickle

# 'audio+image' or 'image-only' or 'audio-only'
CONDITION = 'image-only'  

objective_accuracies = []

all_participant_ids = np.arange(1, 46)
# remove participant 5 due to tehcnical issues (incomplete experiment)
all_participant_ids = all_participant_ids[all_participant_ids != 5]
# remove participant 28 due to missing timestamps
all_participant_ids = all_participant_ids[all_participant_ids != 28]
# remove participant 19 due to technical issues (problematic physiological data)
all_participant_ids = all_participant_ids[all_participant_ids != 19]

feat_names_all_participants = {}
selector_all_participants = {}

for iteration in np.arange(5):

    delta_start = -5
    duration = 24

    for pid in all_participant_ids:
        p = Participant(participantID=pid, rawData=True)
        X = []
        y = []

        ibi_segments = []
        eda_segments = []
        skt_segments = []
        condition_mask = p.markers['condition'] == CONDITION
        for block_index in p.markers[condition_mask].index:

            if CONDITION == 'audio-only':
                audio_onset = p.markers['audio_onset'].iloc[block_index]
                # 1 minute trials with 10s intervals between segments
                onsets = np.arange(audio_onset, audio_onset + 60000, 10000)
            else:
                onsets = json.loads(p.markers.iloc[block_index]['image_onsets'])

            for onset in onsets:

                timestamp = onset / 1000  # Convert milliseconds to seconds
                segment = p.get_rawData_from_timestamp(timestamp, delta_start, duration)
                ibi_segments.append(segment['systolicPeaks'])
                eda_segments.append(segment['eda'])
                skt_segments.append(segment['skt'])
                y.append(p.markers.loc[block_index, 'target_emotion'])

        X = extract_features(ibi_segments, eda_segments, skt_segments)
        X = pd.DataFrame(X)

        X = X.drop(columns=X.columns[X.isna().all()]).replace([np.inf, -np.inf], np.nan)
        ### X could contain NaN values in columns like SampEn (1 value from 54)
        features = X.columns
        feat_names_all_participants[pid] = features

        X = SimpleImputer(strategy='most_frequent').fit_transform(X)
        X = MinMaxScaler().fit_transform(X)

        selector = RFECV(RandomForestClassifier(), step=1, cv=LeaveOneOut(), n_jobs=-1).fit(X, y)
        ### Selects the optimal number of features based on Leave-One-Out cross-validation, 
        ### Ideal for small datasets to prevent overfitting

        selector_all_participants[pid] = selector
        print(f'ITERATION {iteration}')
        print(f"Participant {pid}: Optimal number of features: {selector.n_features_}")

        pickle.dump(feat_names_all_participants, 
                    open(f'feat_names_{CONDITION}_iteration_{iteration}.pkl', 'wb'))
        pickle.dump(selector_all_participants, 
                    open(f'selector_{CONDITION}_iteration_{iteration}.pkl', 'wb'))
        