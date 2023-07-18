# Semantic Segmentation of Ship Images

This repository contains a Python script for performing semantic segmentation of ship images using the SegNet neural network architecture. 


## Usage

The script contains several functions for data preprocessing and model building. To train the model, follow these steps:

1. Set the path to the data set and read the data for training and testing.
2. Define a "decoding" function that takes an encoded mask and decodes it into a binary mask.
3. Define a function "masks_as_image" that takes a list of encoded masks and returns a binary mask.
4. Split the data set into training and testing sets.
5. Define the "keras_generator" function, which generates batches of training data for the model. It takes a dataset containing image IDs and corresponding masks and generates batches of images and masks for training.
6. Generate a training data package using "keras_generator".
7. Define the architecture of the Segnet neural network.
8. Train the model on the training data.

To use the trained model to perform semantic segmentation on a new image, run the predict.py script with the path to the input image as an argument. The segmented image will be saved in the same directory as the input image.

## About SegNet

SegNet is a convolutional neural network architecture that is widely used for semantic segmentation tasks. Semantic segmentation involves assigning a label to each pixel in an image, such as determining which pixels belong to the object and which to the background.

The SegNet architecture consists of an encoder network followed by a decoder network. The input layer has the shape (768, 768, 3), which means that the network can accept an RGB image with a resolution of 768 by 768 pixels. The coding part of the network consists of five convolutional layers, where each layer is followed by the ReLU activation function. The first two convolutional layers have 32 filters, the next two have 64 filters, and the last convolutional layer has 512 filters.

The decoder part of the network consists of five layers of upsampling, where each layer is followed by two convolutional layers with ReLU activation functions. The last convolutional layer in the decoder section has one filter, followed by a sigmoid activation function that produces a probability map of the same size as the input image.  

<blockquote class="imgur-embed-pub" lang="en" data-id="5IsrMbN"><a href="https://imgur.com/5IsrMbN">View post on imgur.com</a></blockquote>
