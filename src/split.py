import split_folders
import sys
import os
this_file = os.path.realpath(__file__)
SCRIPT_DIRECTORY = os.path.split(this_file)[0]
ROOT_DIRECTORY = os.path.split(SCRIPT_DIRECTORY)[0]
sys.path.append(ROOT_DIRECTORY)






if __name__ == '__main__':
    split_folders.ratio(os.path.join(ROOT_DIRECTORY, 'data/gray/all'), output=os.path.join(ROOT_DIRECTORY, 'data/gray/split'), seed=1337, ratio=(.8, .1, .1))

