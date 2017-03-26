# Behavioral Cloning

## Introduction

This writeup documents my submission for [Term 1 Project 3](https://github.com/udacity/CarND-Behavioral-Cloning-P3) of Udacity's [Self-Driving Car Nanodegree](https://medium.com/self-driving-cars/term-1-in-depth-on-udacitys-self-driving-car-curriculum-ffcf46af0c08). The goals of the project are to:

- Build a model that predicts steering angles from images taken from the front of a simulated car,
- Test that the model successfully drives around track one without leaving the road.
- (Optional challenge) test that the model successfully drives around a much more difficult track 2 without leaving the road.

The rest of the writeup will cover:

- What is contained in this repository.
- Background to the assignment and prior art to the assignment itself and methods used in my solution.
- Training data visualization and exploration.
- Training data collection and augmentation.

## Submission Contents

-	`model.py`: (new) script to specify and train the final specific model, and to automatically explore hyperparameters and network architectures.
-	`model.h5`: (new) HDF5-encoded model description and weights that is the main output of `model.py`.
-	`util.py`: (new) utility functions for augmenting the training set and pre-processing images for the model.
-	`drive.py`: (pre-existing) an Udacity-provided script that uses my model to steer the car but comes with a simple proportional-integral (PI) controller that attempts to maintain a constant speed for the car, which is important for the hillier track 2.

In order to use this repository you would:

-	Download and run the Udacity car simulator
- 	In a terminal session run:

```
python drive.py model.h5
```

-	In the car simulator choose either track 1 or track 2. `drive.py` then communicates with the car simulator, receiving front-camera images and responding with steering angles.

## Background and Prior Art

The concept for the project is based heavily on [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316) (Bojarski et. al, 2016), where NVIDIA research scientists successfully trained a convolutional neural network (CNN) on raw pixels from front-facing cameras and angles of the steering column. Rather than specifically extract features for lane line detection, which is a topic I encountered in Term 1 Project 1, they successfully used deep learning as a black box that could extract whatever features are necessary to steer correctly.

In this assignment instead of a real car you start off with a car simulation game where you can drive a car around a track and it captures images, the steering column angle, and other information to disk. The images from track 1 look like:

![](images/01_car_view_center.jpg?raw=true)

and from the significantly more difficult track 2 images look like:

![](images/02_car_view_center_track_2.jpg?raw=true)

Just like the NVIDIA research paper, the car simulation provides you with three images per time instant; one from the center and two from additional left and right cameras. I will go into detail about how models can take advantage of these additional images.

My use of [hyperopt](https://jaberg.github.io/hyperopt/) in order to automatically explore the hyperparameter and network architecture space is not novel and indeed there is a simple wrapper library for Keras available as [hyperas](https://github.com/maxpumperla/hyperas). However I found that using hyperopt directly was more intuitive and gave me more flexibility.

## Training Data Visualization and Exploration

Here is a sample of 50 images from the front-facing camera on both tracks:

![](images/03_sample_of_camera_images.png?raw=true)

We can observe the following about the tracks:

- There is both straight driving and turning,
- The car is always driving between two lines, but the lines vary between:
	- double-solid-yellow, on both sides during straight driving and turns
	- single-solid-white, both the right during straight driving and turns
	- single-dashed-white. on the left during straight driving and turns
	- the red-and-white bumpers on both sides during turns
- Some parts of the road are partially or completely obscured by very dark shadows,
- The horizon varies in brightness and color, and sometimes the car is driving up hill and down hill during straight driving and turns.

Since we are attempting to create a regression model to map front-facing camera images to steering angles it's worth exploring the distribution of angles. The steering angle can vary between -25 to +25 degrees, and is mapped onto a -1 to +1 range for you. This is a histogram of the absolute value of steering angles

![](images/04_distribution_of_angles.png?raw=true)

It's clear that there are substantially more examples of straight driving than turning. If we were to train a model directly on the training data there's a risk that we would overfit on straight driving and be unable to turn effectively. 

## Training data collection and augmentation

Keeping in mind the above, and after much experimentation, it turned out that training data collection and augmentation was the most critical factor in creating a successful model, rather than model architecture and hyperparameter tuning.

In order to collect training data:

- I drove twice clockwise and twice counter-clockwise on each of track 1 and track 2.
- After some initial model training and testing, I identified parts of track 1 and track 2 that the model found difficult and collected additional "recovery" data. I would deliberate start the car on the extreme left/right of the road, start recording, and return the car to the middle of the road. By doing so the model would learn to associate driving close to and onto the edges of the road with recovering to the center of the road.

With this base set of training data I also augmented the training data in various important ways:

- For each center image I flipped it left/right and negated the steering angle. This doubles the training set size. See: [model.py lines 159:161](https://github.com/asimihsan/udacity-carnd-t1-p3-behavioral-cloning/blob/master/model.py#L159-L161).
- At each time slice use the left and right camera images by adding or subtracting a `steering_delta` to the steering angle. This not only increases the training set size but also encourages the car to drive in the center of the correct part of the road. See: [model.py lines 151:165](https://github.com/asimihsan/udacity-carnd-t1-p3-behavioral-cloning/blob/master/model.py#L151-L165).
	- I did not know what value of `steering_delta` would produce the best result, so I treated it as a hyperparameter. Later on I discuss my hyperparameter tuning strategy. In contrast, in the NVIDIA paper they use trigonometry and precise camera placement to determine the best steering delta to apply.
- For each image available to me I increase the training set size and reduce the risk of overfitting by applying various transforms in combination. In isolation, where the original image is in the top-left, they are:
    - random brightness changes, by converting the image's color space to [L\*a\*b](https://en.wikipedia.org/wiki/Lab_color_space#CIELAB) and then multiplying the L-channel by a random factor between 0 and 2. See [util.py lines 8:13](https://github.com/asimihsan/udacity-carnd-t1-p3-behavioral-cloning/blob/master/util.py#L8-L13).

		![](images/05_brightness_augmentation.png?raw=true)    

	- random shadows, by drawing random quadrilaterals sticking out from each side of the image. See: [util.py lines 24:58](https://github.com/asimihsan/udacity-carnd-t1-p3-behavioral-cloning/blob/master/util.py#L24-L58)

		![](images/06_shadow_augmentation.png?raw=true)
		
	-	random shear with border replication (although more subtle this was still a vital augmentation method). See: [util.py lines 71 to 89](https://github.com/asimihsan/udacity-carnd-t1-p3-behavioral-cloning/blob/master/util.py#L71-L89)

		![](images/07_shear_augmentation.png?raw=true)
		
	-	random translations in both x and y up to 30 pixels. The y translations help the model deal with traveling uphill and downhill. The x translations help the model deliver small steering corrections for new scenarios. However how big t