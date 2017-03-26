# Behavioral Cloning

## Introduction

This writeup documents my submission for [Term 1 Project 3](https://github.com/udacity/CarND-Behavioral-Cloning-P3) of Udacity's [Self-Driving Car Nanodegree](https://medium.com/self-driving-cars/term-1-in-depth-on-udacitys-self-driving-car-curriculum-ffcf46af0c08). The goals of the project are to:

- Build a model that predicts steering angles from images taken from the front of a simulated car,
- Test that the model successfully drives around track one without leaving the road.
- (Optionally) test that the model successfully drives around a much more difficult track 2 without leaving the road.

## Background and Prior Art

As part of the assignment you start off with a car simulation game where you can drive a car around a track and it captures images, the steering column angle, and other information to disk. The images look like:

![](images/01_car_view_center.png?raw=true)