# recycling-sorting

In this project, the jetson inference was used to detect is an object can be recycled or not. In our current world, manual trash sorting is, inaccurate, and unsafe in some environments. There is an need for automatic trash sorting using an AI detection trash sorter. With respect to human health and hygiene, as well as the cleanliness of the environment, the effectiveness of garbage sorting is very important. , Many people view trash sorting as a highly time-consuming and difficult process. To overcome this, AI trash sorting detectors would prove to be highly efficient. This project helps solve our current real world problem in a volatile evolving society.

## Overview
In this project, the objects that are supposed to be recycled are detected by imagenet. The project was trained to indentify 2 categories, recyclable and non-recyclable. I used 2 datasets from Kaggler to train my model. These datasets included images of different types of recyclable and non-recyclable objects (plastic bottles, cardboard, aluminum cans, etc.), which I manually categorized. The model is trained to detect how confident it would be that something is recyclable; it is measured on a scale of 0% to 500% to maximize the specification.

## Algorithm
1. I collected two datasets from Kaggle: https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2, https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2
2. I downloaded and unzipped the dataset in my VS Code
3. I manually sorted the images into train, val, and test and categorized them into recyclable and non-recyclable
4. I wrote a python code named train.py to train my model which used imagenet
5. I ran 25 epochs
6. I tested the model and its accuracy by running it on my webcam

## Running this project
To run the project use the following code: 

python3 /home/nvidia/jetson-inference/python/examples/imagenet.py --model=/home/nvidia/jetson-inference/python/training/classification/models/recycling_resnet18.onnx --labels=/home/nvidia/jetson-inference/python/training/classification/models/labels.txt --input-blob=input_0 --output-blob=output_0 /dev/video0

This code runs the project on an attatched webcam to detect if an object is recyclable or not


