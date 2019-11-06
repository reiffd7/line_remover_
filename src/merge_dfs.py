import pandas as pd
import glob

import os
import sys
this_file = os.path.realpath(__file__)
SCRIPT_DIRECTORY = os.path.split(this_file)[0]
ROOT_DIRECTORY = os.path.split(SCRIPT_DIRECTORY)[0]
DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'data') 
CLASSIFICATION_DIRECTORY = os.path.join(DATA_DIRECTORY, 'classification')
sys.path.append(ROOT_DIRECTORY)

def merge(all_files):
    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    frame = frame.fillna(-1)
    return frame

if __name__ == '__main__':
    all_files = glob.glob(os.path.join(CLASSIFICATION_DIRECTORY, '*'))
    result = merge(all_files)
    result.to_csv(os.path.join(CLASSIFICATION_DIRECTORY, 'result.csv'), columns = result.columns)
