# Behavioral Cloning

## Introduction

This writeup documents my submission for [Term 1 Project 3](https://github.com/udacity/CarND-Behavioral-Cloning-P3) of Udacity's [Self-Driving Car Nanodegree](https://medium.com/self-driving-cars/term-1-in-depth-on-udacitys-self-driving-car-curriculum-ffcf46af0c08). The goals of the project are to:

- Build a model that predicts steering angles from images taken from the front of a simulated car,
- Test that the model successfully drives around track one without leaving the road.
- (Optional challenge) test that the model successfully drives around a much more difficult track 2 without leaving the road.

The rest of the writeup will cover:

- Videos of the car driving
- Submission contents, what is contained in this repository.
- Background to the assignment and prior art to the assignment itself and methods used in my solution.
- Training data visualization and exploration.
- Training data collection and augmentation.
- Training and validation methodology.
- Architecture selection and hyperparameter tuning
- Final model architecture

## Videos of the car driving

The following were recording on 640x480 resolution at the highest quality setting.

### Track 1 - overhead view

[![](images/10_track_1_youtube.png?raw=true)](https://www.youtube.com/watch?v=yvpT1ITQ97g)

### Track 2 - overhead view

[![](images/10_track_1_youtube.png?raw=true)](https://www.youtube.com/watch?v=InZHrVgjQGQ)

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

### Collection

In order to collect training data:

- I drove twice clockwise and twice counter-clockwise on each of track 1 and track 2.
- After some initial model training and testing, I identified parts of track 1 and track 2 that the model found difficult and collected additional "recovery" data. I would deliberate start the car on the extreme left/right of the road, start recording, and return the car to the middle of the road. By doing so the model would learn to associate driving close to and onto the edges of the road with recovering to the center of the road.

### Sampling to flatten the distribution of angles

In order to reduce the likelihood of overfitting on straight driving I both oversampled under-represented steering angles and undersampled over-represented steering angles. I used a histogram with 25 bins of aboslute steering angles (see above) and over/under sampled towards the average number of samples per bin, up to a maximum factor of 5, to "flatten" the distribution. See [model.py lines 106:125](https://github.com/asimihsan/udacity-carnd-t1-p3-behavioral-cloning/blob/master/model.py#L106-L125). I found that too aggressively undersampling straight angles led to odd driving on straight roads. The distribution of angles afterwards looked like:

![](images/09_flatter_distribution_of_angles.png?raw=true)

### Augmentation

With this base set of training data I also augmented the training data in various important ways:

- For each center image I flipped it left/right and negated the steering angle. This doubles the training set size. See: [model.py lines 159:161](https://github.com/asimihsan/udacity-carnd-t1-p3-behavioral-cloning/blob/master/model.py#L159-L161).
- At each time slice use the left and right camera images by adding or subtracting a `steering_delta` to the steering angle. This not only increases the training set size but also encourages the car to drive in the center of the correct part of the road. See: [model.py lines 151:165](https://github.com/asimihsan/udacity-carnd-t1-p3-behavioral-cloning/blob/master/model.py#L151-L165).
	- I did not know what value of `steering_delta` would produce the best result, so I treated it as a hyperparameter. Later on I discuss my hyperparameter tuning strategy. In contrast, in the NVIDIA paper they use trigonometry and precise camera placement to determine the best steering delta to apply.
- For each image available to me I increase the training set size and reduce the risk of overfitting by applying various transforms in combination. In isolation, where the original image is in the top-left, they are:
    - random brightness changes, by converting the image's color space to [L\*a\*b\*](https://en.wikipedia.org/wiki/Lab_color_space#CIELAB) and then multiplying the L-channel by a random factor between 0 and 2. See [util.py lines 8:13](https://github.com/asimihsan/udacity-carnd-t1-p3-behavioral-cloning/blob/master/util.py#L8-L13).

		![](images/05_brightness_augmentation.png?raw=true)    

	- random shadows, by drawing random quadrilaterals sticking out from each side of the image. See: [util.py lines 24:58](https://github.com/asimihsan/udacity-carnd-t1-p3-behavioral-cloning/blob/master/util.py#L24-L58)

		![](images/06_shadow_augmentation.png?raw=true)
		
	-	random shear with border replication (although more subtle this was still a vital augmentation method). See: [util.py lines 72 to 90](https://github.com/asimihsan/udacity-carnd-t1-p3-behavioral-cloning/blob/master/util.py#L72-L90)

		![](images/07_shear_augmentation.png?raw=true)
		
	-	random translations in both x and y up to 30 pixels without border replication. The y translations help the model deal with traveling uphill and downhill. The x translations help the model deliver small steering corrections for new scenarios. However how big of a `translation_delta` to apply per pixel of x translation was unknown and I treated it as a hyperparameter to tune for. See [util.py lines 61:69](https://github.com/asimihsan/udacity-carnd-t1-p3-behavioral-cloning/blob/master/util.py#L61-L69) and [model.py lines 167:173](https://github.com/asimihsan/udacity-carnd-t1-p3-behavioral-cloning/blob/master/model.py#L167-L173)

		![](images/08_translation_augmentation.png?raw=true)

### Preprocessing

For all center images used in training and driving I preprocessed them (see [model.py lines 133:158](https://github.com/asimihsan/udacity-carnd-t1-p3-behavioral-cloning/blob/master/util.py#L133-L158)) by:

- Converting their colorspace to [L\*a\*b\*](https://en.wikipedia.org/wiki/Lab_color_space#CIELAB). This helps during training because perceptually similar colors are closer in coordinate space, and moreover the lightness is separated out as a distinct channel and contains the most relevant information for this task.
- Cropping some number of pixels from the top of the image, as not all of the horizon is always relevant to driving. However when driving downhill sometimes the top-most pixels are useful. Hence how many top pixels to crop became a hyperparameter to tune.
- Resizing the image to 128 x 128 pixels, to reduce training time without affecting driving performance too much.
- Normalizing the range of the values of pixels between -0.5 to +0.5, to decrease training time without affecting driving performance.

I tried to use [CLAHE equalization](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization#Contrast_Limited_AHE) to normalize the brightness range of images, but this caused very bad driving performance on track 2 so I excluded it from preprocessing.

## Training and validation methodology

I randomly shuffled the training set and split off 10% of it for validation purposes, and kept 90% for training. Moreover I used a feature of Keras call `fit_generator` which allowed me to process the ~50k images in the training set in batches of 50 images using Python generators. This allows me to use the very large data set of images without having to load all the images and their augmentations into memory. See [model.py lines 274:284](https://github.com/asimihsan/udacity-carnd-t1-p3-behavioral-cloning/blob/master/model.py#L274-L284), [model.py lines 324:335](https://github.com/asimihsan/udacity-carnd-t1-p3-behavioral-cloning/blob/master/model.py#L324-L335), and [model.py lines 134:190](https://github.com/asimihsan/udacity-carnd-t1-p3-behavioral-cloning/blob/master/model.py#L134-L190).

However, as others in the Udacity nanodegree observed, the validation loss during training doesn't correlate very well with driving performance. In addition to only picking models that minimized loss on the held-out validation set, I also stored all models trained at each epoch, and manually tried them all out in the simulator. I observed that the best models tended to get generated between epochs 5-10, and additional training didn't improve driving performance that much. I used an Adam optimizer with a smaller-than-default learning rate of `2e-4` as I observed this improved driving performance.

## Architecture selection and hyperparameter tuning

Not only are there a wide variety of possible deep neural network architectures to attempt, but moreover there are many non-architecture-related hyperparameters that need tuning, as I've outlined above. Rather than rely completely on my intution I used a Python package called [hyperopt](https://jaberg.github.io/hyperopt/), and a space search algorithm called [Tree of Parzen Estimators (TPE)](http://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf) (Bergstra et. al 2011), to perform a search of all possible combinations of architectures and hyperparameters in a way that is more efficient than brute force. I would say this approach is very effective but also prone to local minima, and a lot of iterations with intuition and experimentation is required as well.

At a high level the promise of `hyperopt` is that you can specify any arbitrary function with arguments that returns some loss as a result you want to minimize, describe how to vary those arguments, and it will start exploring it using a method that approximates variables as trees of Gaussian Mixture Models (GMMs). You can see my description of these hyperparameters in [model.py lines 359 to 449](https://github.com/asimihsan/udacity-carnd-t1-p3-behavioral-cloning/blob/master/model.py#L359-L449). By running `hyperopt` over these hyperparameters, and in each iteration minimizing the validation loss, I developed an intuition for the best values for each hyperparameter. Here is a summary:

| Variable | Description | Search range | Chosen value | Notes |
|:---------|:------------|:-------------|:-------------|:----------| 
| `crop_top` | How many pixels to crop from the top of the image, out of 160 pixels | `[0, 5, 10, ..., 70]` | `30` | Values between 20 and 70 seemed good. Cropping at all is better than not cropping. |
| `steering_delta` | How much angle to add for left-camera images, and subtract for right-camera images | `[0.1, 0.0125, 0.0150, ..., 0.4]` | `0.225` | 0.2 to 0.275 seemed good |
| `translation_delta` | For images translated in the x-axis, how much to multiply each pixel of x-translation by to add to the steering angle | `[0.004, 0.005, 0.006, ..., 0.010]` | `0.007` | |
| `use_initial_scaling` | Whether to use an initial 1x1 convolution with 3 feature maps at the start of the CNN | `[True, False]` | `False` | Rather than get the CNN to guess what color space transformation to use I explicitly chose L\*a\*b\*. Using  this initial layer significantly increases training time for not much benefit |
| `conv_activation` | What activation function to use in between convolutional layers | `['relu', 'elu', 'prelu']` | `prelu` | I excluded `srelu` from the search space because it doubled training time. |
| `conv_dropout` | What value of dropout to use between convolutional layers | `[0.0, 0.1, 0.2, ..., 1.0]` | `0.1` | `hyperopt` didn't find any preference for this value, so I used my intution |
| `flatten_activation` | What activation function to use after the flatten layer | `['relu', 'elu', 'prelu']` | `prelu` | I excluded `srelu` from the search space because it doubled training time. |
| `flatten_dropout` | What value of dropout to use after the flatten layer | `[0.0, 0.1, 0.2, ..., 1.0]` | `0.2` | `hyperopt` didn't find any preference for this value, so I used my intution |
| `dense_activation` | What activation function to use after each dense layer | `['relu', 'elu', 'prelu']` | `prelu` | I excluded `srelu` from the search space because it doubled training time. |
| `dense_dropout` | What value of dropout to use after each dense layer | `[0.0, 0.1, 0.2, ..., 1.0]` | `0.2` | `hyperopt` didn't find any preference for this value, so I used my intution |
| `conv_filters` | What numbers of features maps to use for each convolutional layer | (see below) | `[64, 96, 128, 160]` | |
| `conv_kernels` | What kernel sizes to use for each convolutional layer | (see below) | `[7, 5, 3, 3]` | |
| `max_pools` | What pooling size to use for max pooling at each convolutional layer | (see below) | `[3, 2, 2, 2]` | |
| `fc_depths` | How many layers and what sizes to use for the fully-connected layers | `[[512], [512, 512], [512, 512, 512]]` | `[[512]]` | I was curious if more than one fully-connected layer would help, but I observed that a single fully connected layer is best | 

In order to settle on values for `conv_filters` and `conv_kernels` I tried various configurations inspired by two architectures:

- The NVIDIA architecture from their paper (5 convolutional layers, use convolutional strides rather than pooling, 3 fully connected layers)
- The [comma.ai architecture](https://github.com/commaai/research/blob/master/train_steering_model.py) (3 convolutional layers, use convolutional strides rather than pooling, 1 fully connected layer)

I ended up trying many permutations of 4 convolutional layers and 1 fully connected layer and I decided to keep max pooling as a way of allowing the model to learn translational invariance, which I thought would help it drive uphill and downhill on track 2.

Moreover `hyperopt` seemed unable to decide if any kind of dropout was useful or not. Based on my intuition that dropout would have reduce overfitting for the flatten and dense layers, but that perhaps maxpooling was enough to reduce overfitting for the convolutional layers, I arbitrarily chose the values above.

Here are some general observations about using `hyperopt` in this way:

-	I initially tried to optimize the kernel sizes and feature map sizes of convolutions and number of nodes in the fully-connected layers by representing these quantities are `hyperopt.hp.quniform`, i.e. uniformly-distributed values. I wasn't successful at all, even after ensuring that e.g. the number of feature maps increased for each layer of the convolutions, perhaps because of the size of the search space. I found that it's more fruitful to explicitly use `hyperopt.hp.choice` and provide a selection of e.g. 10 combinations to try out.
- `hyperopt` is hard coded to make 20 random guesses before starting optimization based on the suggestion engine. In order to override this you need to do a little hack to set a value for `n_startup_jobs`. See [model.py lines 434:440](https://github.com/asimihsan/udacity-carnd-t1-p3-behavioral-cloning/blob/master/model.py#L434-L440).

## Final model architecture

| Layer | Description |
|:------|:------------|
| Convolution 7x7x64 | 7x7 kernel, 64 feature maps, valid padding |
| PReLu activation | |
| Maxpool 3x3 | |
| Dropout 0.1 | Suppress 10% of nodes during training |
|  | |
| Convolution 5x5x96 | 5x5 kernel, 96 feature maps, valid padding |
| PReLu activation | |
| Maxpool 2x2 | |
| Dropout 0.1 | Suppress 10% of nodes during training |
|  | |
| Convolution 3x3x128 | 3x3 kernel, 128 feature maps, valid padding |
| PReLu activation | |
| Maxpool 3x3 | |
| Dropout 0.1 | Suppress 10% of nodes during training |
|  | |
| Convolution 3x3x160 | 3x3 kernel, 160 feature maps, valid padding |
| PReLu activation | |
| Maxpool 3x3 | |
| Dropout 0.1 | Suppress 10% of nodes during training |
|  | |
| Flatten | |
| Dropout 0.2 | Suppress 20% of nodes during training |
| PReLu activation | |
|  | |
| Fully-connected 512 | |
| Dropout 0.2 | Suppress 20% of nodes during training |
| PReLu activation | |
|  | |
| Single output node | |

See [model.py lines 451:464](https://github.com/asimihsan/udacity-carnd-t1-p3-behavioral-cloning/blob/master/model.py#L451-L464) (the parameters) and [model.py lines 206:248](https://github.com/asimihsan/udacity-carnd-t1-p3-behavioral-cloning/blob/master/model.py#L206-L248) (how the model is created using the parameters).