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
- Training data augmentation.

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

## Training data augmentation

