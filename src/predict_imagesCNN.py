from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from CNN import imageCNN
from standardizer import Standardizer
from imageData_generator import ImageGenerator
import glob
import os
import sys
from skimage import io, color, filters, feature, restoration
this_file = os.path.realpath(__file__)
SCRIPT_DIRECTORY = os.path.split(this_file)[0]
ROOT_DIRECTORY = os.path.split(SCRIPT_DIRECTORY)[0]
DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'data') 
CLASSIFICATION_DIRECTORY = os.path.join(DATA_DIRECTORY, 'classification/merged')
RESULTS_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'results') 
MODELS_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'models/models')
sys.path.append(ROOT_DIRECTORY)


class LineScrubber(object):

    def __init__(self, gray_image, threshold, whitespace, model_path, figname):
        self.gray_image = gray_image.copy()
        self.img_rows = gray_image.shape[0]
        self.img_cols = gray_image.shape[1]
        self.threshold = threshold
        self.whitespace = whitespace
        self.model = load_model(model_path)
        self.figname = figname
        self.scrub()

    def plot_frame(self, zoom, row_index, col_index, size):
        masked_window = np.random.random((zoom.shape[0],zoom.shape[1]))
        masked_window[row_index:row_index+size, col_index:col_index+size] = 1
        masked_window = np.ma.masked_where(masked_window != 1, masked_window)

        masked_pixel = np.random.random((zoom[row_index:row_index+size, col_index:col_index+size].shape[0],zoom[row_index:row_index+size, col_index:col_index+size].shape[1]))
        masked_pixel[15,15] = 1
        masked_pixel = np.ma.masked_where(masked_pixel != 1, masked_pixel)

        masked_pixel1 = np.random.random((zoom[row_index:row_index+size, col_index:col_index+size].shape[0],zoom[row_index:row_index+size, col_index:col_index+size].shape[1]))
        masked_pixel1[8:23, 15] = 1
        masked_pixel1 = np.ma.masked_where(masked_pixel1 != 1, masked_pixel1)
        
        window = zoom[row_index:row_index+size, col_index:col_index+size]
        colored_percentage = np.count_nonzero(window==0)/(30**2)
        pixel_value = window[15, 15]
        window_sobel = ndimage.sobel(window, axis=0)
        above_area = np.mean(window_sobel[8:16, 15])
        below_area = np.mean(window_sobel[16:23, 15])
        sobel_value = window_sobel[15, 15]
        
        # Overlay the two images
        fig, ax = plt.subplots(1, 3)
        ax.ravel()
        ax[0].imshow(zoom, cmap='gray')
        ax[0].imshow(masked_window, cmap='prism', interpolation='none')
        # ax[0].imshow(masked_pixel, cmap=cm.jet, interpolation='none')
        ax[0].set_title('Colored: {}'.format(round(colored_percentage, 2)))
        ax[1].imshow(window, cmap='gray')
        ax[1].imshow(masked_pixel, cmap='prism', interpolation='none')
        ax[1].set_title('Pixel Value: {}'.format(pixel_value))
        ax[2].imshow(window_sobel)
        ax[2].imshow(masked_pixel1, cmap='jet')
        ax[2].imshow(masked_pixel, cmap='prism', interpolation='none')

    def predict(self, gray_window):
        X = gray_window
        X = X.reshape(-1, 30, 30, 1)
        # print(X.shape)
        prob = self.model.predict(X)[0][0]
        print(prob)
        if prob >= self.threshold:
            prediction = 1
        else:
            prediction = 0
        
        return prediction

    def alter_image(self, i, j, prediction):
        if prediction == 1:
            self.gray_image[i+15, j+15] = self.whitespace ## mean of whitespace
            print('pixel changed')

    def save_fig(self, path):
        plt.imsave(path, self.gray_image,  cmap='gray')

    def scrub(self, size=30):
        gray = self.gray_image
        visit_list = np.argwhere(gray <= (self.whitespace - 5))
        self.save_fig(os.path.join(RESULTS_DIRECTORY, '{}_before.png'.format(self.figname)))
        for x in visit_list:
            i = x[0]-15
            j = x[1]-15
            print(i, j)
            # self.plot_frame(gray, i, j, size)
            # self.plot_frame(binar, i, j, size)
            # plt.show()
            gray_window = gray[i:i+size, j:j+size]
            prediction = self.predict(gray_window)
            print(prediction)
            self.alter_image(i, j, prediction)
        self.save_fig(os.path.join(RESULTS_DIRECTORY, '{}_after.png'.format(self.figname)))


if __name__ == '__main__':
    print('Loading model')
    model_path = '../models/models/CNN_E100_Batch10_Filters64_Neurons64_Actrelu_Layers_3.h5'

    print('Loading resized images')
    resized_imgs = glob.glob('../data/medium/*')

    print('Subset the images that we want, the ones we trained the model on')
    img_list = [164, 202, 425, 345, 139, 72, 311, 363, 403, 509, 362, 257, 175, 203, 47, 183, 0, 297, 34, 8, 320, 197, 293, 450, 215, 28, 74]
    img_subset = []
    for img in resized_imgs:
        img_idx = int(img.split('/')[3].split('_')[0])
        if img_idx in img_list:
            img_subset.append(img)

    print('Standardize the images')
    standardizer_subset = Standardizer(img_subset, resized_imgs[3])

    print('Select the image we want to scrub')
    bin_image = standardizer_subset.binarized_images[6]
    grey_image = standardizer_subset.greyscale_image_list[6]
    img_name = standardizer_subset.image_list[6].split('/')[3].split('.')[0]
    print(img_name)

    m_type = 'CNN'
    
    i = 3

    print('Ready to scrub')
    images = ImageGenerator(bin_image, grey_image, img_name)
    images.pad(15, 237.4)
    gray = images.gray_padded_image

    thresholds = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    
    scrubber = LineScrubber(gray, 0.55, 202.0, model_path, '{}_{}_{}_test'.format(img_name, 'CNN_E100_Batch10_Filters64_Neurons64_Actrelu_Layers_3.h5', 0.55))
