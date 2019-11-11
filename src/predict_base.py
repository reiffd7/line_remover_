from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np 
from skimage.transform import resize, rotate
from skimage.color import rgb2gray
from scipy.spatial.distance import squareform
import pickle
from scipy.misc import imread
from scipy import ndimage
import pandas as pd
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



class Predicter(object):

    def __init__(self, gray_image, threshold, whitespace, model_path, figname):
        self.gray_image = gray_image.copy()
        self.img_rows = gray_image.shape[0]
        self.img_cols = gray_image.shape[1]
        self.threshold = threshold
        self.whitespace = whitespace
        self.model_path = model_path
        self.figname = figname



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

    def alter_image(self, i, j, prediction):
        if prediction == 1:
            self.gray_image[i+15, j+15] = self.whitespace ## mean of whitespace
            print('pixel changed')

    def save_fig(self, path):
        plt.imsave(path, self.gray_image,  cmap='gray')


    def predict(self):
        pass

    def scrub(self):
        pass



class CNNScrubber(Predicter):
    '''
    The class is fed a grayscale image, a prediction threshold, a whitespace pixel intensity value (grayscale), 
    a CNN model path (to be loaded), and a figure name. With this information, we iterate through dark pixels (whitespace - 10), 
    and make a prediction with the loaded model based on the 30x30 array of pixels surrounding the dark pixel at each iteration. 
    If the prediction is a line, the grayscale image pixel value becomes the whitespace pixel intensity value. Before and after images 
    are saved. 
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = load_model(self.model_path)
        self.scrub()

    
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


class ClassifierScrubber(Predicter):

    '''
    The class is fed a binarized and grayscale image corresponding to the same original image,
    a prediction threshold, a whitespace pixel intensity value (grayscale), a classification model path (to be pickled),
    and a figure name. With this information, we iterate through dark pixels (whitespace - 10), collect 
    associated features, and make a prediction based on the loaded model. If the prediction is a line,
    the grayscale image pixel value becomes the whitespace pixel intensity value. Before and after images 
    are saved. 
    '''

    def __init__(self, bin_image, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bin_image = bin_image.copy()
        self.model = pickle.load(open(self.model_path, 'rb'))
        self.scrub()

    

    def predict(self, gray_pixel_value, mean_pixel_value, colored_percentage, sobel_gradient, last_3):
        normalized_pixel_val = ((self.whitespace - gray_pixel_value)/self.whitespace)
        normalized_mean_pixel_val = ((self.whitespace - mean_pixel_value)/self.whitespace)
        X = np.array([[gray_pixel_value, mean_pixel_value, colored_percentage, sobel_gradient, last_3[0], last_3[1], last_3[2], normalized_pixel_val, normalized_mean_pixel_val]])
        X = X.reshape((1, -1))
        # print(X.shape)
        prob = self.model.predict_proba(X)[0][1]
        print(prob)
        if prob >= self.threshold:
            prediction = 1
        else:
            prediction = 0
        return prediction

   
    def scrub(self, size=30):
        gray = self.gray_image
        binar = self.bin_image
        visit_list = np.argwhere(gray <= (self.whitespace - 10))
        last_3 = [-1, -1, -1]
        # self.save_fig(os.path.join(RESULTS_DIRECTORY, '{}_before.png'.format(self.figname)))
        for x in visit_list:
            i = x[0]-15
            j = x[1]-15
            print(i, j)
            # self.plot_frame(gray, i, j, size)
            # self.plot_frame(binar, i, j, size)
            # plt.show()
            gray_window = gray[i:i+size, j:j+size]
            bin_window = binar[i:i+size, j:j+size]
            gray_window_sobel = ndimage.sobel(gray_window, axis=0)
            mean_pixel_value = np.mean(gray_window)
            gray_pixel_value = gray_window[15, 15]
            colored_percentage = np.count_nonzero(bin_window==0)/(30**2)
            above_area = np.mean(gray_window_sobel[8:16, 15])
            below_area = np.mean(gray_window_sobel[16:23, 15])
            sobel_gradient = above_area - -below_area
            # print('Pix Val: {}, Mean Pix Val: {}, Bin Colored: {}, Sobel Gradient: {}, Last 3: {}'.format(gray_pixel_value, mean_pixel_value, colored_percentage, sobel_gradient, last_3))
            prediction = self.predict(gray_pixel_value, mean_pixel_value, colored_percentage, sobel_gradient, last_3)
            print(prediction)
            self.alter_image(i, j, prediction)
            last_3.pop()
            last_3.insert(0, prediction)
        self.save_fig()


if __name__ == '__main__':
    print('Loading model')
    model_path = '../models/models/moredata_CNN_E100_Batch10_Filters64_Neurons64_Actrelu_Layers_3.h5'

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
   
    bin_image = standardizer_subset.binarized_images[2]
    grey_image = standardizer_subset.greyscale_image_list[2]
    img_name = standardizer_subset.image_list[2].split('/')[3].split('.')[0]
    print(img_name)
    arr = np.array(grey_image)
    first10_flat = arr[:10, :].flatten()
    n, bins, patches = plt.hist(first10_flat, bins=30)
    bin_max = np.where(n == n.max())
    whitespace = bins[bin_max][0]
    


    print('Ready to scrub')
    images = ImageGenerator(bin_image, grey_image, img_name)
    images.pad(15, whitespace)
    gray = images.gray_padded_image

    thresholds = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    
    scrubber = CNNScrubber(gray, 0.55, whitespace, model_path, '{}_{}_{}_test'.format(img_name, 'moredata_CNN_E100_Batch10_Filters64_Neurons64_Actrelu_Layers_3.h5', 0.55))
