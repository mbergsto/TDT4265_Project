# TDT4265_Project

# Project Title: Object and Keypoint Detection System

## Overview
This project implements an advanced detection system focusing on both object detection and keypoint detection within images. The primary application is for analyzing sports imagery, such as football matches, to identify objects like the ball and players, and to detect specific keypoints that are crucial for sports analytics.

## Features
- **Object Detection**: Utilizes YOLO (You Only Look Once) models to identify and locate various objects in an image.
- **Keypoint Detection**: Employs advanced deep learning models to detect and map keypoints related to players and other relevant objects.
- **Performance Metrics**: Provides detailed analysis like precision, recall, and mean Average Precision (mAP) metrics for both objects and keypoints across different sets of images.

## Getting Started
These instructions will help you set up the project locally for development and testing purposes.

### Prerequisites
- Python 3.8+
- pip

### Installation
1. **Clone the repository**
git clone https://github.com/mbergsto/TDT4265_Project.git


2. **Setup a virtual environment** 
python -m venv .venv
source .venv/bin/activate


3. **Install the required packages**
pip install -r requirements.txt


### Usage
Run the main scripts to start the detection process:
python trainer.ipynb (Object detection and tracker)
python keypoint_detection.ipynb (keypoint detection)


## Configuration
- Update `xxx` with model paths, thresholds, and other necessary parameters to fine-tune the object detection.

- Update `botsort.yaml` to change parameters for tracking.