# Line Remover - Applied to Uncle Peter's Sailboats
A project using CNNs to remove ruled lines from sketches

![logo](/presentation_imgs/logo.png)


1. [Overview](#overview)
2. [The Data](#the-data)
2. [The Strategy](#the-strategy)
3. [The Pipeline](#the-pipeline)
4. [Data Classification](#data-classification)
5. [CNN Architecture](#cnn-architecture)
5. [CNN Parameters](#cnn-parameters)
5. [CNN Results](#cnn-results)
6. [Picture Scrubbing](#picture-scrubbing)
7. [Results](#results)
7. [Comparison](#comparison)
8. [App](#app)
8. [Further Work](#further-work)
10. [Sources](#sources)



## Overview


Thank you to Land Belenky for the project idea. Land's Uncle Peter accumulated 1124 sailboat pencil sketches over the years. Unfortunately, 513 of these images were done on ruled paper. Can we salvage the ruled pictures?

Hypothesis: It is possible to train a CNN to remove ruled lines from an image without apparent degradation of the image


Goals

<li> Classify whether or not the central pixel of an individual frame of a picture belongs to a drawing or a line </li>
<li> Train a CNN to predict whether or not a pixel belongs to a line</li>
<li> Use the trained CNN to scrub a picture of its lines</li>
<li> Maintain the quality of the image throughout the process </li>


## The Data

Ruled | Unruled
:-------------------------:|:-------------------------: 
![ruled](/presentation_imgs/ruled_EX.jpg)  |  ![unruled](/presentation_imgs/unruled_EX1.jpg)


1124 total sailboat drawings, 513 on ruled paper. Some files are corrupt. Taking a closer look at just the ruled drawings ... 

Dimensions |
:-----------
![dimensions](/presentation_imgs/EDA1.png)


Aspect Ratio |
:-----------
![dimensions](/presentation_imgs/EDA2.png)


It is clear that the drawings will have to be standardized

<li>The dimensions of the drawings are very large and widely distributed. The aspect ratios are centered around ~1.25, so we will use this to resize each image to 400x502.</li>
<li>The drawings will be converted to grayscale</li>
<li>Pixel intensities of the drawings will be normalized</li>

![greyscale](/presentation_imgs/eq.gif) 

![standardize](/presentation_imgs/standardize.png) 


## The Strategy

At first, I considered augmenting lines onto unruled images. I would then train an autoencoder to remove lines from these images. Finally, I would apply the model to ruled images.

Due to the difference in quality across the images, I could find a process by which to augment lines in a way that would be similar to ruled images. I would also have to resize these very large images, degrading the quality.

I found a new strategy from a paper that showed how it was possible to remove staff lines from music scores. Instead of looking at the entire picture, this paper looked at 28 x 28 windows, classified if the central pixel came from a staff or symbol. A CNN was then trained on staff and symbol classifications. Staff classification were then removed. The results were impressive. 

![inspiration](/presentation_imgs/inspiration.png) 


## The Pipeline

![pipeline](/presentation_imgs/Line_Remover.png) 

From a directory of standardized ruled sailboat drawings, we classify thousands of 30x30 frames and split them into train and test directories. The CNN is trained and tested using these directories. Finally, we take a standardized sailboat drawing to be scrubbed. The drawing is divided into 30x30 frames. We predict whether or not each frame is a line or a drawing frame. The lines frames are removed. The result is a scrubbed sailboat drawing.

## Data Classification 

Line Class | Drawing Class
:-------------------------:|:-------------------------: 
![ruled](/presentation_imgs/line_class.png)  |  ![unruled](/presentation_imgs/drawing_class.png)

Data is needed to train, test, and predict on a Convolutional Neural Network. I took a sample of 27 drawings and iterated through sections of each drawing, labeling frames as lines or drawings depending on the location of the central pixel. After initially collecting 7,000 total frames (around half/half line/drawing), I looked at the mean pixel intensities for each class for each drawing. I wanted to make sure we were representing all different kinds of lines and drawings that appear on sailboat drawings. 

-- | --
:-------------------------:|:-------------------------: 
![ruled](/presentation_imgs/framesEDA1.png)  |  ![unruled](/presentation_imgs/framesEDA2.png)

As you can see, lines tend to be very similar no matter what drawing they are sampled from. Drawings, on the other hand, occur in many different forms across each drawing. 


## CNN Architecture

![cnn](/presentation_imgs/CNN.png) 

CNN architecture was determined through a process of trial and error. I would create a model, scrub an image, and look at the results hoping to maximize line removal and drawing preservation. 

**Observations**

<li> Increasing layers from 4 to 6 aided the process of line removal</li>
<li> Increasing epochs from 2 to 5 to 10 helped preserve the drawing</li>
<li> Increasing number of filters from 2 to 8 helped line removal. Increasing filters from 8 to 16 to 32 to 64 helped preserve the drawings. Further increasing number of filters to 128 tarnished the drawings </li>
<li> Number of neurons adjustment had a very similar effect to number of filters</li>



## CNN Parameters

| Parameter           | Value   |
|---------------------|---------|
| Epochs              | 10     |
| Batch Size          | 10      |
| Image Size          | 30 x 30 |
| Filters             | 64      |
| Neurons             | 64      |
| Layers              | 6       |
| Kernel Size         | 4 x 4   |
| Pool Size           | 2 x 2   |
| Activation Function | Relu    |
| Optimizer | Adam    |

<br>
<br>
<br>

 Adjusting Epochs | 
 | :-------------------------:
![epochs](/presentation_imgs/epochs1.gif)  | 

 Adjusting Filters | 
 | :-------------------------:
![epochs](/presentation_imgs/filters2.gif)  | 

 Adjusting Layers | 
 | :-------------------------:
![epochs](/presentation_imgs/layers.gif)  | 


## CNN Results


 Accuracy | 
 | :-------------------------:
![acc](/presentation_imgs/CNN_acc.png)  | 

 Precision/Recall | 
 | :-------------------------:
![PR](/presentation_imgs/CNN_PR.png) | 

 Loss | 
 | :-------------------------:
![Loss](/presentation_imgs/CNN_loss.png) | 


Based on training data, the model is very accurate with great precision/recall rates. You can also clearly see in these graphs that after 10 epochs, test/training accuracy, precision, recall, and loss are not further improved. It is also aparent, especially in loss, that the model starts overfitting to the train data. These observations are confirmed in the scrubbed results. Drawings are just as well preserved at 10 epochs as they are at 100 epochs. 


## Picture Scrubbing

For the final step, I used my model to predict frame by frame, whether or not the central pixel is from a line or a drawing. If the pixel is from a line, the pixel is removed. 

This step is extremely computational expensive. To cut down on cost, I adopted a couple strategies:

<li> Drawings are orginally reshaped to 400x502 </li>
<li> Pixels are only visited if they are less than 10 units darker than whitespace, or the background. This cut down number of pixels visited by 95%</li>
<li> Each frame is not saved. Rather, each frame is represented in a numpy array which is reshaped and predicted as either a drawing or a line</li>

 To speed up the process, I used a memory-optimized EC2 instance. Overall, I sped up the process from 2 hours to 1 minute and 10 seconds. 


 ## Results

| Before           | After  |
|---------------------|---------|
| ![r1_A](/presentation_imgs/r1_B.png)             | ![r1_B](/presentation_imgs/r1_A.png)     |
| ![r2_B](/presentation_imgs/r2_B.png)           | ![r1_A](/presentation_imgs/r2_A.png)       |
| ![r1_A](/presentation_imgs/r3_B.png)          | ![r1_A](/presentation_imgs/r3_A.png)  |
| ![r1_A](/presentation_imgs/r4_B.png)              | ![r1_A](/presentation_imgs/r4_A.png)       |



## Comparison

![comparison](/presentation_imgs/comparison.png) 

The CNN appears to work better to other line removal strategies. 

<li> The Fourier Transfrom strategy looks at the image in FT space where you can only see lines due to their frequency. Lines are removed in FT space. This strategy appears to need extremely consistent lines and more filtering or pieces of lines will remain

<li> For the Random Forest model to work, each frame is featurized by flattening 30x30 frames to a vector of length of 900. The Random Forest removes lines well but completely destorys the integrity of the drawing


## App

![app](/presentation_imgs/home_screen.png) 
![cnn](/presentation_imgs/preview.png) 
![cnn](/presentation_imgs/scrub.png) 

## Further Work

**More Data**
<li> The Model performs beautifully on my results, however it does not work on every sailboat or any drawing on ruled paper</li>
<li> The Model needs to be trained on thousands more frames of lines and drawings to be deployable in an app </li>


**Multiprocessing**
<li> As it is now, the app can scrub an image in around 1 minute and 10 seconds</li>
<li> To further speed this up, I will need to utilize multiprocessing</li>
<li> Instead of visiting only dark pixels, I will have to visit all pixels, but do this in parallel</li> 
<li>Assign rows to different cores using multiprocessing</li>

**Colorizing**
<li> I converted theses drawings to grayscale which was OK for this project because most sketches were done in pencil</li>
<li> But for this to be deployable on a wider scale, I want to colorize grayscale images if the original sketches were done in color</li>
<li> Colorization could also be utilized to create art from pencil sketches </li>
<li> GANs could be utilized to capture the styles of similar paintings/pictures </li>