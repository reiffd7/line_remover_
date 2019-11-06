import numpy as np
import matplotlib.pyplot as plt
import glob


from skimage import io, color, filters, feature, restoration
from skimage.transform import resize, rotate
from skimage.color import rgb2gray
from scipy.spatial.distance import squareform
from scipy.misc import imread
import matplotlib.cm as cm
from scipy import ndimage, misc


class standardizer(object):
    '''
    This class takes in a list of RGB images. It has the capability to
    create a list of greyscale images. From the greyscale images, we standardize
    each image and then binarize each image. The class also can plot as many 
    images as we want in a grid
    '''

    def __init__(self, image_list, image):
        self.image_list = image_list
        self.image = image


    def greyscale(self, n):
        greyscale_images = []
        for i in range(n):
            try:
                image = imread(self.image_list[i], mode = 'L')
                greyscale_images.append(image)
            except:
                continue
        self.greyscale_image_list = np.array(greyscale_images)

    def greyscale_one(self):
        self.greyscale_image = imread(self.image, mode = 'L')

    def standardize(self):
        standardized_image_list = []
        for image in self.greyscale_image_list:
            standardized_image = (image - np.min(image))/(np.max(image) - np.min(image))
            standardized_image_list.append(standardized_image)
        self.standardized_image_list = np.array(standardized_image_list)
    
    def standardize_one(self):
        self.standardized_image = (self.greyscale_image - np.min(self.greyscale_image))/(np.max(self.greyscale_image) - np.min(self.greyscale_image))

    def binarize(self, threshold):
        binarized_images = []
        for image in self.standardized_image_list:
            binarized = 1.0 * (image > threshold)
            binarized_images.append(binarized)
        self.binarized_images = np.array(binarized_images)

    def binarize_one(self, threshold):
        self.binarized_image = 1.0* (self.standardized_image > threshold)

    def visualize(self, rows, cols):
        fig, ax = plt.subplots(rows, cols, figsize = (20, 10))
        for i, a in enumerate(ax.flatten()):
            a.imshow(self.binarized_images[i], cmap='gray')
        plt.show()




if __name__ == '__main__':
    # unruled = glob.glob('../Sketches/Unruled/*')
    ruled = glob.glob('../Sketches/Ruled/*')

    images_30 = standardizer(ruled)
    images_30.greyscale(30)
    images_30.standardize()
    images_30.binarize(0.7)
    images_30.visualize(6, 4)
