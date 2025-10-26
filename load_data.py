import preprocessing
import segmentation
from avro.datafile import DataFileReader
from avro.io import DatumReader
import os
import json

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def load_avro_subject(dir_path: str):
    """
    Input: path of the directory containing the avro files of the subject
    Returns a list containing: acc, bvp, eda, skt, systolicPeaks, tags
    The signals are contained in pd.DataFrames, except for systolicPeajs and tags 
    which are stored in pd.Series with the corresponding timestamps
    """
    # read avro files; data is a list of rawData readings
    data = []
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        with open(file_path, 'rb') as avro_file:
            reader = DataFileReader(avro_file, DatumReader())
            schema = json.loads(reader.meta.get('avro.schema').decode('utf-8'))
            data.append(next(reader))
    
    acc = pd.DataFrame()
    acc_x = pd.Series()
    acc_y = pd.Series()
    acc_z = pd.Series()
    Fs_acc = data[0]['rawData']['accelerometer']['samplingFrequency']
    if Fs_acc == 0.0:
        Fs_acc = 64
    dt_acc = 1.0 / Fs_acc
    start_acc = data[0]['rawData']['accelerometer']['timestampStart'] / 1000000.0

    bvp = pd.DataFrame()
    bvp_value = pd.Series()
    Fs_bvp = data[0]['rawData']['bvp']['samplingFrequency']
    if Fs_bvp == 0.0:
        Fs_bvp = 64
    dt_bvp = 1.0 / Fs_bvp
    start_bvp = data[0]['rawData']['bvp']['timestampStart'] / 1000000.0

    eda = pd.DataFrame()
    eda_value = pd.Series()
    Fs_eda = data[0]['rawData']['eda']['samplingFrequency']
    if Fs_eda == 0.0:
        Fs_eda = 4
    dt_eda = 1.0 / Fs_eda
    start_eda = data[0]['rawData']['eda']['timestampStart'] / 1000000.0

    skt = pd.DataFrame()
    skt_value = pd.Series()
    Fs_skt = data[0]['rawData']['temperature']['samplingFrequency']
    if Fs_skt == 0.0:
        Fs_skt = 1
    dt_skt = 1.0 / Fs_skt
    start_skt = data[0]['rawData']['temperature']['timestampStart'] / 1000000.0

    systolicPeaks = pd.Series()

    tags = pd.Series()

    #aCC
    for block in data:
        acc_x = pd.concat([acc_x, 
                           pd.Series(block['rawData']['accelerometer']['x'])
                           ])
        acc_y = pd.concat([acc_y, 
                           pd.Series(block['rawData']['accelerometer']['y'])
                           ])
        acc_z = pd.concat([acc_z, 
                           pd.Series(block['rawData']['accelerometer']['z'])
                           ])
        bvp_value = pd.concat([
            bvp_value, pd.Series(block['rawData']['bvp']['values'])
        ])
        eda_value = pd.concat([
            eda_value, pd.Series(block['rawData']['eda']['values'])
        ])
        skt_value = pd.concat([
            skt_value, pd.Series(block['rawData']['temperature']['values'])
        ])
        systolicPeaks = pd.concat([
            systolicPeaks, pd.Series(block['rawData']['systolicPeaks']['peaksTimeNanos'])
        ])
        tags = pd.concat([
            tags, pd.Series(block['rawData']['tags']['tagsTimeMicros'])
        ])
        
    acc['x'] = acc_x
    acc['y'] = acc_y
    acc['z'] = acc_z 
    acc['timestamp'] = float(start_acc) + (np.arange(len(acc)) * dt_acc)
    acc = acc.reset_index(drop=True)

    bvp['value'] = bvp_value
    bvp['timestamp'] = float(start_bvp) + (np.arange(len(bvp)) * dt_bvp)
    bvp = bvp.reset_index(drop=True)

    eda['value'] = eda_value
    eda['timestamp'] = float(start_eda) + (np.arange(len(eda)) * dt_eda)
    eda = eda.reset_index(drop=True)

    skt['value'] = skt_value
    skt['timestamp'] = float(start_skt) + (np.arange(len(skt)) * dt_skt)
    skt = skt.reset_index(drop=True)

    systolicPeaks = systolicPeaks / 1000000000.0
    systolicPeaks = systolicPeaks.reset_index(drop=True)

    tags = tags / 1000000.0
    tags = tags.reset_index(drop=True)

    subject_data = {
        'acc': acc,
        'bvp': bvp,
        'eda': eda,
        'skt': skt,
        'systolicPeaks': systolicPeaks,
        'tags': tags,
        'Fs_acc': Fs_acc,
        'Fs_bvp': Fs_bvp,
        'Fs_eda': Fs_eda,
        'Fs_skt': Fs_skt,
        'rawData': data[0]['rawData']
    }

    return subject_data

def load_participant(file):
    """ 
    Returns a List of pd.DataFrames containing the signals information with timestamp
    order: acc, bvp, eda, temp, hr, ibi
    """
    aCC = pd.read_csv(file+'\aCC.csv')
    acc = pd.DataFrame()
    acc[['x', 'y', 'z']] =  aCC[1:]
    dt_acc = 1.0 / 32
    acc['timestamp'] = float(aCC.columns[0]) + (np.arange(len(acc)) * dt_acc) # first row of the csv file is unix timestamp

    bVP = pd.read_csv(file+'\bVP.csv')
    bvp = pd.DataFrame()
    bvp['value'] = bVP[1:]
    dt_bvp = 1.0 / 64
    bvp['timestamp'] = float(bVP.columns[0]) + (np.arange(len(bvp)) * dt_bvp)

    eDA = pd.read_csv(file+'\eDA.csv')
    eda = pd.DataFrame()
    eda['value'] = eDA[1:]
    dt_eda = 1.0 / 4
    eda['timestamp'] = float(eDA.columns[0]) + (np.arange(len(eda)) * dt_eda)
    
    tEMP = pd.read_csv(file+'\tEMP.csv')
    temp = pd.DataFrame()
    temp['value'] = tEMP[1:]
    dt_temp = 1.0 / 4
    temp['timestamp'] = float(tEMP.columns[0]) + (np.arange(len(temp)) * dt_temp)

    hR = pd.read_csv(file+'\hR.csv')
    hr = pd.DataFrame()
    hr['value'] = hR[1:]
    dt_hr = 1.0
    hr['timestamp'] = float(hR.columns[0]) + (np.arange(len(hr)) * dt_hr)

    iBI = pd.read_csv(file+'\iBI.csv')
    ibi = pd.DataFrame()
    ibi[['timestamp', 'value']] = iBI
    ibi['timestamp'] = ibi['timestamp'] + float(iBI.columns[0])

    return acc, bvp, eda, temp, hr, ibi