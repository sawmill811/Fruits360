# Fruits360
## Introduction
This is our course project for CSL2050 Pattern Recognition and Machine Learning (2022 Winter Semester). The <u>problem statement</u> is: To recognise different fruits and vegetables from the given dataset using Machine Learning techniques. 

## Dataset
The dataset we used is called Fruits360 - a dataset with 90380 images of 131 fruits and vegetables. The images are of dimensions 100px by 100px and are in RGB format.

## Pipeline
Our task in this project was to build an end-to-end pipeline to classify the fruits and vegetables in the dataset. Our pipeline has the following functionalities:
* Loads pretrained model from GitHub
* Loads given image and processes it
* Predicts the class of given image

The pipeline has 7 methods – 2 public and 5 private. We can print the summary of the pipeline model using the summary() method. We can make predictions using the predict(image_path) method. While making a prediction, the input image is shown along with the model’s prediction.
We also integrated this Pipeline in the website that we built for this project. The website could not be deployed because of limited hosting resources.

## Running the pipeline
### Prerequisites
You must have the following packages installed in your working environment:
* `Python 3.8.0`
* `numpy==1.22.3`
* `Pillow==9.1.0`
* `Flask==2.1.1`
* `keras==2.8.0`
* `tensorflow==2.8.0`

Next, Use `git clone https://github.com/sawmill811/Fruits360.git` in your terminal to clone the repository.
<br>
Type `flask run` in your terminal and go to http://localhost:5000/ to see the website. Ignore the TensorFlow warnings, if any. You should be able to see something like this:
<br>
<img src="https://user-images.githubusercontent.com/90380/84598981-f9c8c580-f9c9-11ea-9c0c-e9f9c8d8f8b9.png" alt="Fruits360" width="500">







