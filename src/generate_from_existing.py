from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage import io, color, filters, feature, restoration
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
this_file = os.path.realpath(__file__)
SCRIPT_DIRECTORY = os.path.split(this_file)[0]
ROOT_DIRECTORY = os.path.split(SCRIPT_DIRECTORY)[0]
DOCUMENTS = os.path.split(ROOT_DIRECTORY)[0]
DATA_STASH = os.path.join(DOCUMENTS, 'line_remover_2_data_stash')
GRAY_STASH = os.path.join(DATA_STASH, 'gray')
BINARY_STASH = os.path.join(DATA_STASH, 'binar')
sys.path.append(ROOT_DIRECTORY)


class imageGenerator(object):
    '''
    This class takes a path to a directory of images, the directory we want to
    export images to, and the number of images we want to create. With that 
    information, we generate new images that are our original images but skewed,
    zoomed, rotated, etc. The new images are saved into our export directory.
    '''
    

    def __init__(self, image, export_path, n):
        self.image = image
        self.export_path = export_path
        self.n = n
        self.create_and_save_images()

    def create_img_generator(self):
        # ''' this creates the image generator object'''
        self.datagen = ImageDataGenerator(
                        # rotation_range=60,
                        horizontal_flip=True,
                        fill_mode='nearest')
        print('generator created')


    def create_pictures(self):
        # '''This will run through the generator created
        # in the create_img_generator function to actually save the images'''
        img = load_img(self.image)
        x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

        # the .flow() command below generates batches of randomly transformed images
        # and saves the results to the `preview/` directory
        i = 0
        for batch in self.datagen.flow(x, batch_size=1,
                                save_to_dir=self.export_path, save_prefix=self.image, save_format='png'):
            i += 1
            if i > self.n:
                break  # otherwise the generator would loop indefinitely

    def create_and_save_images(self):
        self.create_img_generator()
        self.create_pictures()




if __name__ == '__main__':
    print('Hey')
    imageGenerator(os.path.join(GRAY_STASH, 'all/lines/509_img_190_348.png'), 
                    os.path.join(GRAY_STASH, 'all/lines'), 200)

    # imageGenerator(os.path.join(ROOT_DIRECTORY, 'drawings'), 
    #                 os.path.join(ROOT_DIRECTORY, 'generated_drawings'), 100)


    