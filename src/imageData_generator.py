import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io, color, filters, feature, restoration
from skimage.transform import resize, rotate
from skimage.color import rgb2gray
from scipy.spatial.distance import squareform
import matplotlib.cm as cm
from scipy import ndimage, misc
import os
import sys
this_file = os.path.realpath(__file__)
SCRIPT_DIRECTORY = os.path.split(this_file)[0]
ROOT_DIRECTORY = os.path.split(SCRIPT_DIRECTORY)[0]
DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'data') 
# RESIZED_IMAGES_DIRECTORY = os.path.join(DATA_DIRECTORY, 'medium')
# BINARY_DIRECTORY = os.path.join(DATA_DIRECTORY, 'binar')
# GRAYSCALE_DIRECTORY = os.path.join(DATA_DIRECTORY, 'gray')
CLASSIFICATION_DIRECTORY = os.path.join(DATA_DIRECTORY, 'classification')
DOCUMENTS = os.path.split(ROOT_DIRECTORY)[0]
DATA_STASH = os.path.join(DOCUMENTS, 'line_remover_2_data_stash')
GRAY_STASH = os.path.join(DATA_STASH, 'gray')
BINARY_STASH = os.path.join(DATA_STASH, 'binar')
sys.path.append(ROOT_DIRECTORY)


class ImageGenerator(object):

    def __init__(self, bin_image, gray_image, name):
        self.bin_image = bin_image
        self.gray_image = gray_image
        self.name = name

    def pad(self, n, shade):
        result_g = np.full((self.gray_image.shape[0]+2*n, self.gray_image.shape[1]+2*n), shade)
        result_g[n:self.gray_image.shape[0]+n, n:self.gray_image.shape[1]+n] = self.gray_image
        self.gray_padded_image = result_g
        result = np.ones((self.bin_image.shape[0]+2*n, self.bin_image.shape[1]+2*n))
        result[n:self.bin_image.shape[0]+n, n:self.bin_image.shape[1]+n] = self.bin_image
        self.bin_padded_image = result


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
    #     return above_area, below_area, colored_percentage

    def get_label(self):
        label = input()
        if label == 'l':
            label_for_entry = 1 ## 1 is for LINE
        elif label == '':
            label_for_entry = 1
        else:
            label_for_entry = 0 ## 0 is for DRAWING
        return label_for_entry

    def update_dict(self, gray_pixel_value, mean_pixel_value, colored_percentage, sobel_gradient, label):
        self.my_dict['gray_pixel_value'].append(gray_pixel_value)
        self.my_dict['mean_gray_pixel_value'].append(mean_pixel_value)
        self.my_dict['bin_percentage_colored'].append(colored_percentage)
        self.my_dict['sobel_gradient'].append(sobel_gradient)
        self.my_dict['label'].append(label)

    def save_imgs(self, label, gray, binary, row_idx, col_idx):
        binary_lines_path = os.path.join(BINARY_STASH, 'all/lines')
        binary_drawings_path = os.path.join(BINARY_STASH, 'all/drawings')
        gray_lines_path = os.path.join(GRAY_STASH, 'all/lines')
        gray_drawings_path = os.path.join(GRAY_STASH, 'all/drawings')
        if label == 1:
            plt.imsave(os.path.join(binary_lines_path, '{}_{}_{}.png'.format(self.name, row_idx, col_idx)), binary, cmap='gray')
            plt.imsave(os.path.join(gray_lines_path, '{}_{}_{}.png'.format(self.name, row_idx, col_idx)), gray, cmap='gray')
        else:
            plt.imsave(os.path.join(binary_drawings_path, '{}_{}_{}.png'.format(self.name, row_idx, col_idx)), binary, cmap='gray')
            plt.imsave(os.path.join(gray_drawings_path, '{}_{}_{}.png'.format(self.name, row_idx, col_idx)), gray, cmap='gray')

    def add_label_history(self):
        self.df['label_t1'] = ""
        self.df['label_t2'] = ""
        self.df['label_t3'] = ""
        for idx, row in self.df.iterrows():
            if idx > 0:
                self.df.at[idx,'label_t1'] = list(self.df[self.df.index == idx - 1].label)[0]
                if idx > 1:
                    self.df.at[idx,'label_t2'] = list(self.df[self.df.index == idx - 2].label)[0]
                    if idx > 2:
                        self.df.at[idx,'label_t3'] = list(self.df[self.df.index == idx - 3].label)[0]
    
    def save_csv(self, row_idx):
        self.df.to_csv(os.path.join(CLASSIFICATION_DIRECTORY, '{}_{}.csv').format(self.name, row_idx), columns=self.df.columns)

    def gen_data(self, row, size=30):
        gray = self.gray_padded_image
        binar = self.bin_padded_image
        self.my_dict = {'gray_pixel_value': [], 'mean_gray_pixel_value': [], 'bin_percentage_colored': [], 'sobel_gradient': [], 'label': []}
        for r in range(30, 220):
            row_idx = r
            for c in range(230, 255):
                col_idx = c
                self.plot_frame(gray, row_idx, col_idx, size)
                self.plot_frame(binar, row_idx, col_idx, size)
                plt.show()
                gray_window = self.gray_padded_image[row_idx:row_idx+size, col_idx:col_idx+size]
                bin_window = self.bin_padded_image[row_idx:row_idx+size, col_idx:col_idx+size]
                gray_pixel_value = gray_window[15, 15]
                bin_pixel_value = bin_window[15, 15]
                mean_pixel_value = np.mean(gray_window)
                gray_window_sobel = ndimage.sobel(gray_window, axis=0)
                colored_percentage = np.count_nonzero(bin_window==0)/(30**2)
                above_area = np.mean(gray_window_sobel[8:16, 15])
                below_area = np.mean(gray_window_sobel[16:23, 15])
                sobel_gradient = above_area - -below_area
                if bin_pixel_value == 0.0:
                    label = self.get_label()
                    self.update_dict(gray_pixel_value, mean_pixel_value, colored_percentage, sobel_gradient, label)
                    self.save_imgs(label, gray_window, bin_window, row_idx, col_idx)
                    print(label)


        self.df = pd.DataFrame.from_dict(self.my_dict)
        self.add_label_history()
        self.save_csv(row_idx)
