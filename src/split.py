import split_folders
import sys
import os
this_file = os.path.realpath(__file__)
SCRIPT_DIRECTORY = os.path.split(this_file)[0]
ROOT_DIRECTORY = os.path.split(SCRIPT_DIRECTORY)[0]
DOCUMENTS = os.path.split(ROOT_DIRECTORY)[0]
DATA_STASH = os.path.join(DOCUMENTS, 'line_remover_2_data_stash')
GRAY_STASH = os.path.join(DATA_STASH, 'gray/all')
BINARY_STASH = os.path.join(DATA_STASH, 'binary/all')
sys.path.append(ROOT_DIRECTORY)






# if __name__ == '__main__':
#     split_folders.ratio(BINARY_STASH, output=os.path.join(ROOT_DIRECTORY, 'data/gray/split'), seed=1337, ratio=(.8, .1, .1))

