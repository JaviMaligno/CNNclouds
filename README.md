# CNNclouds
A convolutional neural network that aims to distinguish three types of clouds.

## Introduction

This little project was inspired by the Coursera project called [Classify Images of Clouds in the Cloud with AutoML Vision](https://www.coursera.org/projects/googlecloud-classify-images-of-clouds-in-the-cloud-with-automl-vision-boydv?utm_medium=coursera&utm_source=promo). The aim of this project is to classify cloud images according to three types of clouds. The Coursera project used AutoML Vision, which is already made and ready to use (and has excellent performance). This means that there is little insight or space for exploration.

For this reason, I decided to try and develop a Convolutional Neural Network (CNN) of my own using the same dataset that I downloaded from Coursera. To do this I followed [Keras official documentation](https://keras.io/examples/vision/image_classification_from_scratch/) on image classification from scratch. I also got some ideas from [Datacamp](#https://www.datacamp.com/tutorial/convolutional-neural-networks-python) and googling around.

## Description

The training set, contained in the 'cloud' folder, is made of three folders, each for a type of cloud: cirrus, cumulonimbus and cumulus. Each of these folders contains 20 images, making a total of 60 images (this is a rather small dataset). I augmented the train dataset via rotations. Below there is an example of each type of cloud.


  <p align = "center">
<img  src="https://user-images.githubusercontent.com/25660622/210348196-90e8d6cb-aac7-44eb-a046-26c8fcc2dd9c.jpg" alt="Cirrus" height = "250" width = "250">  
<img src="https://user-images.githubusercontent.com/25660622/210349163-1e0570a6-f160-44af-9150-618c809cb6b8.jpg" alt="Cumulonimbus" height = "250" width = "250">
  <img src="https://user-images.githubusercontent.com/25660622/210352894-1cefd65e-5520-48f1-9cfb-9911af2a41bb.jpg" alt="Cumulus" height = "250" width = "250">
  </p>
  <br>
  <p>
Fig.1 - Cirrus, Cumulonimbus, Cumulus.</figcaption>
  </p>

  
Since the training set is small, I used a batch size of 1. My intention was to use no batch size at all, but Keras models require a batch size. 

Before I started creating the model I noticed that the test images were given as JSON files, in which the image was encoded as a base64 string in one of the fields. Therefore, I wrote a function called `tensor_image` to decode the image and transform it into a tensor that Keras could manipulate. 

The neural network is designed in the function `make_model`. It is a simplification of the one given by Keras documentation above, in the sense that it has no residuals and only uses softmax as final activation. The model is also adapted to my data set: my image have size 256x256 and there are 3 labels (correspoding to the 3 types of clouds). 

It mainly consists of separable convolution and relu activation layers. Finally it has a global average pooling layer so that the output has only one dimension and a dense layer to make this dimension match the number of layers. A plot of the model can be seen below. 

<p align = "center">
<img src="https://user-images.githubusercontent.com/25660622/210354800-427188ba-1fc6-4347-9a49-121491b2aa27.png" alt="model" width = "400" height = "2000">
  </p>


## Results 

The model I have uploaded is just the final one that I decided to keep in the end, but I have made several experiments tweaking different parameters. For instance, I tried different number of filters for each convolution layer. Changing the dropout also varies the performance but in a non-linear way, I found that it works better around 0.25. However I kept the number of epochs stable.

Despite all these tweaks, the CNN performs very poorly, failing even on the train set and not being able to recognize the test images, which are given below.

<p align = "center">
<img src="https://user-images.githubusercontent.com/25660622/210355465-373fbef5-ee3d-4f82-ad46-05f98383a04b.png" alt="Test 1" height = "400" width = "400">
<img src="https://user-images.githubusercontent.com/25660622/210355648-d44df4bc-4e1b-49dd-8c6e-dbbd8ea1b019.png" alt="Test 2" width = "400" height = "400">
</p>
<p>
Fig.2 - Tests: Cirrus, Cumulonimbus.</figcaption>
  </p>

For some reason, I found that whenever the CNN classified well the test images it would perform very poorly on the training set. This was usually correlated to greater dropout, which would explain worse performance on training set but not the better performance on test. However, the test is so small that we shouldn't infer anything from it. 

Overall, my model is very bad.
