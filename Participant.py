import os
import json
import pandas as pd
import params as p
from load_data import load_avro_subject


class Participant:
    """
    A class to represent a single participant's data
    This class must be in the same folder as markers.csv

    Attributes:
        participantID (int or str): A unique identifier for the participant.

        demographics (dict)
        age (int): The age of the participant in years.
        gender (str): The self-reported gender of the participant.
        education (str): The highest level of education attained by the participant.
        field_of_study (str): The current field of study the participant is involved with.

        questionnaires (dict)
        dif_score (int): Difficulty identifying feelings (Associated items: 1, 3, 6, 7, 9, 13, 14)
        ddf_score (int): Difficulty describing feelings	(Associated items: 2, 4, 11, 12, 17)
        eot_score (int): Externally-oriented thinking	(Associated items: 5, 8, 10, 15, 16, 18, 19, 20)
        tas_score (int or float): The participant's score on the Toronto Alexithymia Scale (TAS-20).

        aq_score (int or float): The participant's score on the Autism Spectrum Quotient (AQ).

        panas_score (dict): A dictionary containing the participant's scores on the
            Positive and Negative Affect Schedule (PANAS), e.g., {'positive': 20, 'negative': 15}.
        pa_score (int or float): Positive Affect (items 1, 3, 5, 9, 10, 12, 14, 16, 17, and 19). 
        na_score (int or float): Negative Affect Sore (items 2, 4, 6, 7, 8, 11, 13, 15, 18, and 20). 

        markers (pandas.DataFrame): A DataFrame containing time-stamped experimental
            markers, loaded from 'markers.csv'.
        rawData (dict): A dictionary of raw physiological signals.
            Keys are strings representing the signal name, and values are array-like
            (e.g., list, pandas.Series, or numpy.ndarray).
            Expected keys include:
            - 'acc': 3-axis accelerometer data.
            - 'bvp': Blood Volume Pulse, for heart rate analysis.
            - 'eda': Electrodermal Activity (galvanic skin response).
            - 'skt': Skin Temperature.
            - 'systolicPeaks': Timestamps or indices of systolic peaks detected in the BVP signal.
    """
    def __init__(self, participantID: int, rawData=False):
        """
        Initializes the Participant object and loads all data.

        Args:
            participantID (int): Identifier of the participant.
        """
        self.participantID = participantID

        behavioral_data_path = os.path.join(p.DATA_PATH, f'participant_{participantID}')

        demographics_path = os.path.join(behavioral_data_path, p.DEMOGRAPHICS_FILE_PREFIX+f'{participantID:02}'+p.DEMOGRAPHICS_FILE_SUFFIX)
        with open(demographics_path, 'r') as file:
            self.demographics = json.load(file)
        for key in self.demographics.keys():
            setattr(self, key, self.demographics[key])

        questionnaires_path = os.path.join(behavioral_data_path, p.QUESTIONNAIRE_FILE_PREFIX+f'{participantID:02}'+p.QUESTIONNAIRE_FILE_SUFFIX)
        with open(questionnaires_path, 'r') as file:
            self.questionnaires = json.load(file)

        dif_items = [1, 3, 6, 7, 9, 13, 14]
        dif_score = 0
        for item in dif_items:
            dif_score = dif_score + self.questionnaires['tas20']['responses'][item-1]
        self.dif_score = dif_score
        ddf_items = [2, 4, 11, 12, 17]
        ddf_score = 0
        for item in ddf_items:
            ddf_score = ddf_score + self.questionnaires['tas20']['responses'][item-1]
        self.ddf_score = ddf_score
        eot_items = [5, 8, 10, 15, 16, 18, 19, 20]
        eot_score = 0
        for item in eot_items:
            eot_score = eot_score + self.questionnaires['tas20']['responses'][item-1]
        self.eot_score = eot_score
        self.tas_score = dif_score + ddf_score + eot_score

        pa_items = [1, 3, 5, 9, 10, 12, 14, 16, 17, 19]
        pa_score = 0
        for item in pa_items:
            pa_score = pa_score + self.questionnaires['panas']['responses'][item-1]
        self.pa_score = pa_score
        na_items = [2, 4, 6, 7, 8, 11, 13, 15, 18, 20]
        na_score = 0
        for item in na_items:
            na_score = na_score + self.questionnaires['panas']['responses'][item-1]
        self.na_score = na_score
        self.panas_score = pa_score + na_score

        markers_path = os.path.join(behavioral_data_path, p.MARKERS_FILE_PREFIX+f'{participantID:02}'+p.MARKERS_FILE_SUFFIX)
        
        self.markers = pd.read_csv(markers_path)
        if rawData:
            self.rawData = load_avro_subject(os.path.join(p.RAW_DATA_PATH, f'participant_{participantID}'))

    def __str__(self):
        """Returns a comprehensive, readable string representation of the participant's data."""
        demo_details = '\n'.join([f"    - {key.capitalize()}: {value}" for key, value in self.demographics.items()])
        marker_summary = f"DataFrame with {self.markers.shape[0]} rows and {self.markers.shape[1]} columns"

        return (
            f"===========================================\n"
            f"| Participant Data Summary: ID {self.participantID}     |\n"
            f"===========================================\n\n"
            f"--- Demographics ---\n"
            f"{demo_details}\n\n"
            f"--- TAS-20 Scores ---\n"
            f"  - Difficulty Identifying Feelings (DIF): {self.dif_score}\n"
            f"  - Difficulty Describing Feelings (DDF): {self.ddf_score}\n"
            f"  - Externally-Oriented Thinking (EOT):  {self.eot_score}\n"
            f"  - TOTAL TAS Score:                       {self.tas_score}\n\n"
            f"--- PANAS Scores ---\n"
            f"  - Positive Affect (PA): {self.pa_score}\n"
            f"  - Negative Affect (NA): {self.na_score}\n\n"
            f"--- Experimental Data ---\n"
            f"  - Markers: {marker_summary}\n"
            f"-------------------------------------------"
        )

    def get_indexes(self):
        return list(self.markers.index)
    
    # def get_rawData_from_indexes(self, index_list, delta_start, delta_end):
    #     """
    #     """

    def get_rawData_from_indexes(self, index_list, delta_start, duration):
        """Extracts segments of raw signal data for a given list of event indexes.

        This method iterates through a list of marker indexes, and for each one,
        it extracts a slice of all available raw signals. The slice is
        defined by a time window relative to the event's timestamp.

        Args:
            index_list (list of int): A list of indexes from the self.markers DataFrame.
            delta_start (float or int): The start time of the segment in seconds, relative
                to the event's timestamp (e.g., -5 to start 5 seconds before).
            duration (float or int): The end time of the segment in seconds, relative
                to the event's timestamp (e.g., 60 to end 60 seconds after).

        Returns:
            dict: A dictionary where keys are the integer indexes from index_list.
                  Each value is another dictionary containing the segmented raw signals
                  (e.g., {'bvp': pd.Series(...), 'eda': pd.Series(...)}).
        """
        all_segments = {}

        for index in index_list:
            # Get the anchor timestamp for the current event index
            event_timestamp = self.get_index_timestamp(index)
            
            # Calculate the absolute start and end timestamps for the segment
            segment_start_ts = event_timestamp + delta_start
            segment_end_ts = segment_start_ts + duration
            
            # This will store the segmented signals for the current index
            single_event_segments = {}
            
            # Iterate over all available raw data signals
            for signal_name, signal_data in self.rawData.items():
                # Check if the signal is a DataFrame with a 'timestamp' column
                if isinstance(signal_data, pd.DataFrame) and 'timestamp' in signal_data.columns:
                    mask = (signal_data['timestamp'] >= segment_start_ts) & \
                           (signal_data['timestamp'] <= segment_end_ts)
                    single_event_segments[signal_name] = signal_data[mask]
                
                # Check if the signal is a Series (assumed to be timestamps)
                elif isinstance(signal_data, pd.Series):
                    mask = (signal_data >= segment_start_ts) & (signal_data <= segment_end_ts)
                    single_event_segments[signal_name] = signal_data[mask]

            all_segments[index] = single_event_segments
            
        return all_segments

    def get_rawData_from_timestamp(self, timestamp, delta_start, duration):
        segment_start_ts = timestamp + delta_start
        segment_end_ts = segment_start_ts + duration

        rawData_segments = {}
        for signal_name, signal_data in self.rawData.items():
            # Check if the signal is a DataFrame with a 'timestamp' column
            if isinstance(signal_data, pd.DataFrame) and 'timestamp' in signal_data.columns:
                mask = (signal_data['timestamp'] >= segment_start_ts) & \
                        (signal_data['timestamp'] <= segment_end_ts)
                rawData_segments[signal_name] = signal_data[mask]
            
            # Check if the signal is a Series (assumed to be timestamps)
            elif isinstance(signal_data, pd.Series):
                mask = (signal_data >= segment_start_ts) & (signal_data <= segment_end_ts)
                rawData_segments[signal_name] = signal_data[mask]

        return rawData_segments

    def get_timestamp_from_index(self, index, img_idx = 0):
        i_markers = self.markers.loc[index]
        if i_markers['condition'] != 'image-only':
            return float(i_markers['audio_onset'])/1000
        else:
            return float(json.loads(i_markers['image_onsets'])[img_idx])/1000
    
    def emotion_identification_mask(self, signal_name, index):
        if isinstance(self.signals[signal_name], pd.DataFrame):
            return (self.signals[signal_name]['timestamp'] >= self.get_index_timestamp(index) + 60) & \
                (self.signals[signal_name]['timestamp'] <= self.get_index_timestamp(index) + 85)
        return (self.signals[signal_name] >= self.get_index_timestamp(index) + 60) & \
               (self.signals[signal_name] <= self.get_index_timestamp(index) + 85)
    
    def get_emotion_identification(self, index):
        signals_segment = {}
        for signal_name in self.signal_names:
            mask = self.emotion_identification_mask(signal_name, index)
            signals_segment[signal_name] = self.signals[signal_name][mask]
        return signals_segment



    def get_bvp(self, index):
        mask = (self.signals['bvp']['timestamp'] >= self.get_index_timestamp(index)) & \
               (self.signals['bvp']['timestamp'] <= self.get_index_timestamp(index) + 60)
        return self.signals['bvp'][mask]

    def get_ibi(self, index):
        mask = (self.signals['systolicPeaks'] >= self.get_index_timestamp(index)) & \
               (self.signals['systolicPeaks'] <= self.get_index_timestamp(index) + 60)
        return self.signals['systolicPeaks'][mask] * 1000
    
    def get_ibi_segments(self, index, WD, SP):
        windows = []
        startTS = self.get_index_timestamp(index)
        endTS = startTS + 60
        while True:
            mask = (self.signals['systolicPeaks'] >= startTS) & \
                   (self.signals['systolicPeaks'] <= (startTS + WD))
            windows.append(self.signals['systolicPeaks'][mask] * 1000)
            startTS = startTS + WD + WD*SP
            if startTS+WD > endTS:
                break
        return windows

    def get_eda_segments(self, index, WD, SP):
        windows = []
        startTS = self.get_index_timestamp(index)
        endTS = startTS + 60
        while True:
            mask = (self.signals['eda']['timestamp'] >= startTS) & \
                   (self.signals['eda']['timestamp'] <= (startTS + WD))
            windows.append(self.signals['eda'][mask])
            startTS = startTS + WD + WD*SP
            if startTS+WD > endTS:
                break
        return windows
    
    def get_skt_segments(self, index, WD, SP):
        windows = []
        startTS = self.get_index_timestamp(index)
        endTS = startTS + 60
        while True:
            mask = (self.signals['skt']['timestamp'] >= startTS) & \
                   (self.signals['skt']['timestamp'] <= (startTS + WD))
            windows.append(self.signals['skt'][mask])
            startTS = startTS + WD + WD*SP
            if startTS+WD > endTS:
                break
        return windows

    def get_eda(self, index):
        mask = (self.signals['eda']['timestamp'] >= self.get_index_timestamp(index)) & \
               (self.signals['eda']['timestamp'] <= self.get_index_timestamp(index) + 60)
        return self.signals['eda'][mask]

    def get_skt(self, index):
        mask = (self.signals['skt']['timestamp'] >= self.get_index_timestamp(index)) & \
               (self.signals['skt']['timestamp'] <= self.get_index_timestamp(index) + 60)
        return self.signals['skt'][mask]

# p1 = Participant(signals_path=r'C:\Users\--\Documents\Alexithymia\classification\raw_data\participant_0',
#                  markers_path=r'C:\Users\--\Documents\Alexithymia\data\test_participant\test_participant.csv')

# print(
#     p1.signals.keys(),
#     p1.get_indexes()
# )
p0 = Participant(participantID=3)
print(p0.pa_score)
print(p0.panas_score)
# print(p0.rawData)