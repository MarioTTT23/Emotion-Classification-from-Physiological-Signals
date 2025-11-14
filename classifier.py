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
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


from Participant import Participant
from feature_extraction import skt_features
from helper import extract_segments, extract_features

import pickle

def get_svm_importances(fitted_estimator):
    """
    Returns a 1D array of importances from a fitted SVM
    by taking the L2 norm (Euclidean distance) of the coefficients
    across the classes.
    """
    # Access the model from the pipeline
    svm_model = fitted_estimator.named_steps['svm']
    # Get the 2D coefficient matrix (shape [n_classes, n_features])
    coefs = svm_model.coef_
    # Calculate the L2 norm across the class rows (axis=0)
    # This results in a 1D array of shape [n_features]
    importances = np.linalg.norm(coefs, axis=0)
    return importances

def get_lr_importances(fitted_estimator):
    """
    Returns a 1D array of importances from a fitted LogisticRegression
    by taking the L2 norm (Euclidean distance) of the coefficients
    across the classes.
    """
    lr_model = fitted_estimator.named_steps['lr'] # <-- Accesses the 'lr' step
    coefs = lr_model.coef_
    importances = np.linalg.norm(coefs, axis=0)
    return importances

### Initialize participant IDs
all_participant_ids = np.arange(1, 46)
# remove participant 5 due to tehcnical issues (incomplete experiment)
all_participant_ids = all_participant_ids[all_participant_ids != 5]
# remove participant 28 due to missing timestamps
all_participant_ids = all_participant_ids[all_participant_ids != 28]
# remove participant 19 due to technical issues (problematic physiological data)
all_participant_ids = all_participant_ids[all_participant_ids != 19]

### Initialize storage for classifiers
classifiers = {}

### Window parameters: Signal to Features
delta_start = -5
duration = 24

### Iterate over participants and conditions
for pid in all_participant_ids:
    ### Initialize storage for participant-sepcific information 
    p = Participant(participantID=pid, rawData=True)

    classifiers[pid] = {}
        
    for condition in {'audio+image', 'image-only', 'audio-only'}:

        classifiers[pid][condition] = {}

        ### Initialize storage for feature matrix and labels
        X = []
        y = []

        ### Storage for segments of each physiological signal
        ### Lists of dictionaries
        ibi_segments = []
        eda_segments = []
        skt_segments = []

        ### Iterate over blocks of the specified condition
        condition_mask = p.markers['condition'] == condition
        ## Drop one random sample of each target emotion for 'audio-only' condition
        if condition == 'audio-only':
            audio_only_df = p.markers[p.markers['condition'] == 'audio-only'].copy()
            target_emotions = ['Happiness', 'Anger', 'Sadness']
            filtered_audio_only_df = audio_only_df[audio_only_df['target_emotion'].isin(target_emotions)] 
            try:
                sampled_rows = filtered_audio_only_df.groupby('target_emotion').sample(n=1, random_state=42)
                remaining_rows = audio_only_df.drop(sampled_rows.index)
            except ValueError:
                remaining_rows = audio_only_df # If sampling fails, use all rows
        else:
            remaining_rows = p.markers[condition_mask]
            
        for block_index in remaining_rows.index:

            ### Get onsets depending on condition
            if condition == 'audio-only':
                audio_onset = p.markers['audio_onset'].iloc[block_index]
                # 60 second trials with 10 second intervals between segments
                # Intervals have the same duration as an image presentation
                onsets = np.arange(audio_onset, audio_onset + 60000, 10000)
            else:
                onsets = json.loads(p.markers.iloc[block_index]['image_onsets'])

            ### Extract segments for each onset
            for onset in onsets:
                timestamp = onset / 1000  # Convert milliseconds to seconds
                segment = p.get_rawData_from_timestamp(timestamp, delta_start, duration)
                ibi_segments.append(segment['systolicPeaks'])
                eda_segments.append(segment['eda'])
                skt_segments.append(segment['skt'])
                y.append(p.markers.loc[block_index, 'target_emotion'])


        print(f'Participant {pid}, Condition: {condition}')

        X = extract_features(ibi_segments, eda_segments, skt_segments)
        X = pd.DataFrame(X)

        # Some features are only defined for segments of certain minimal lengths
        # X could contain NaN values in columns like SampEn (1 value from 54)
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.drop(columns=X.columns[X.isna().all()])

        preprocessor = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('remove_constant', VarianceThreshold())
        ])
        preprocessor.set_output(transform="pandas")
        X_processed = preprocessor.fit_transform(X)

        loocv = LeaveOneOut()

        # GaussianNB should NOT be scaled, as it expects Gaussian-distributed data.
        # --- 1. SVM Tuning ---
        pipe_svm = Pipeline([
            ('scale', StandardScaler()),
            ('svm', SVC(random_state=42))
        ])
        # We tune 'C' (regularization) and 'kernel'
        param_svm = {
            'svm__C': [0.1, 1, 10],
            'svm__kernel': ['linear', 'rbf']
        }
        grid_svm = GridSearchCV(pipe_svm, param_svm, cv=loocv, scoring='accuracy', n_jobs=-1)
        grid_svm.fit(X_processed, y)
        classifiers[pid][condition]['grid_svm'] = grid_svm

        # --- 2. KNN Tuning ---
        pipe_knn = Pipeline([
            ('scale', StandardScaler()),
            ('knn', KNeighborsClassifier())
        ])
        # We tune 'n_neighbors' (the 'k' value)
        param_knn = {
            'knn__n_neighbors': [3, 5, 7, 9] # Small, odd numbers
        }
        grid_knn = GridSearchCV(pipe_knn, param_knn, cv=loocv, scoring='accuracy', n_jobs=-1)
        grid_knn.fit(X_processed, y)
        classifiers[pid][condition]['grid_knn'] = grid_knn

        # --- 3. Gaussian Naive Bayes (No Tuning) ---
        # This model has no real hyperparameters, so we just get its score
        pipe_gnb = Pipeline([
            # No scaling!
            ('gnb', GaussianNB())
        ])
        grid_gnb = GridSearchCV(pipe_gnb, {}, cv=loocv, scoring='accuracy', n_jobs=-1) # Empty param grid
        grid_gnb.fit(X_processed, y)
        classifiers[pid][condition]['grid_gnb'] = grid_gnb

        # --- Encode your labels ---
        le = LabelEncoder()
        # This converts ['Happiness', 'Anger', 'Sadness'] into [1, 0, 2]
        y_numeric = le.fit_transform(y)
        classifiers[pid][condition]['label_encoder'] = le

        pipe_nn = Pipeline([
            ('scale', StandardScaler()),
            ('nn', MLPClassifier(max_iter=1000, early_stopping=True, random_state=42))
        ])
        # We tune the size of the single hidden layer
        param_nn = {
            'nn__hidden_layer_sizes': [(10,), (20,), (10, 5)] # (10,) = one layer of 10 neurons
        }
        grid_nn = GridSearchCV(pipe_nn, param_nn, cv=loocv, scoring='accuracy', n_jobs=-1)
        grid_nn.fit(X_processed, y_numeric)
        classifiers[pid][condition]['grid_nn'] = grid_nn

        # --- 5. RandomForest ---
        pipe_rf = Pipeline([
            # No scaling needed
            ('rf', RandomForestClassifier(random_state=42))
        ])
        param_rf = {
            'rf__n_estimators': [50, 100],
            'rf__max_depth': [3, 5, None]
        }
        grid_rf = GridSearchCV(pipe_rf, param_rf, cv=loocv, scoring='accuracy', n_jobs=-1)
        grid_rf.fit(X_processed, y)
        classifiers[pid][condition]['grid_rf'] = grid_rf

        rf_model = RandomForestClassifier(random_state=42)
        rfecv_rf = RFECV(estimator=rf_model, step=1, cv=loocv, scoring='accuracy', n_jobs=-1)
        rfecv_rf.fit(X_processed, y)
        classifiers[pid][condition]['rfecv_rf'] = rfecv_rf

        svc_pipeline = Pipeline([
            ('scale', StandardScaler()),
            ('svm', SVC(kernel='linear', C=1, random_state=42)) 
        ])
        rfecv_svm = RFECV(
            estimator=svc_pipeline, 
            step=1, 
            cv=loocv, 
            scoring='accuracy', 
            n_jobs=-1,
            importance_getter=get_svm_importances 
        )
        rfecv_svm.fit(X_processed, y)
        classifiers[pid][condition]['rfecv_svm'] = rfecv_svm

        lr_pipeline = Pipeline([
            ('scale', StandardScaler()),
            ('lr', LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42))
        ])
        rfecv_lr = RFECV(
            estimator=lr_pipeline, 
            step=1, 
            cv=loocv, 
            scoring='accuracy', 
            n_jobs=-1,
            importance_getter=get_lr_importances 
        )
        rfecv_lr.fit(X_processed, y)
        classifiers[pid][condition]['rfecv_lr'] = rfecv_lr

        pickle.dump(classifiers, 
                    open(f'classifiers_.pkl', 'wb'))
        pickle.dump(X_processed, 
                    open(f'x_processed_.pkl', 'wb'))

