import csv
import numpy as np
import pickle
import sys
import pandas as pd
import cnn


def data_extractor():
    df = pd.read_csv('data/training.csv')
    df = df.dropna()

    labels = df.columns[:-1]

    # Normalize the data

    df['Image'] = df['Image'].apply(
        lambda x: [int(i) / 255.0 for i in x.split(' ')])
    df[labels] = df[labels].apply(lambda x: x / 96.0)

    return df, df.shape[1] - 1
