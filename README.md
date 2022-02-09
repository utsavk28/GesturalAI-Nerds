# GesturalAI

## 📌 About
In this project we created a application that classifies the sign language to english letters.

## Group members
- [Utsav Khatu](https://github.com/utsavk28)
- [Tushar Bauskar](https://github.com/tusharsb-12)

## 🎯 Key Features
* Classify the American Sign Language to english letters in real time by taking video input from the user

## 💻 Development Process

* First we worked on classifying most common words in English from sign language. For this, we used the WLASL dataset and fine tuned the I3D model which was pretrained on the Kinetics-400 dataset
* Then we created our own small dataset for 10 common words. Each word had 30 videos of 30 frames. Then we extracted the landmarks of pose, left hand and right hand using mediapipe holistic model. And then use the landmarks of the frames for training the RNN model. This model has very less accuracy. 
* We then trained different models on ASL dataset (consisting of sign languages of A-Z letters and some special characters like SPACE, DELETE and NOTHING). This model was finally used in our project

### Datasets used
* [WLASL video dataset](https://dxli94.github.io/WLASL/)
  * WLASL is the largest video dataset for Word-Level American Sign Language (ASL) recognition
  * This dataset consists of around 20K videos for sign language representation of around 2000 commonly used words

* [American Sign Language Dataset for letters](https://www.kaggle.com/grassknoted/asl-alphabet)
  * This dataset contains 87000 images of 200 X 200 pixels, divided into 29 classes (A - Z, SPACE, DELETE and NOTHING)
  * The test data set contains only 29 images

### Notebooks
* [ASL Alphabet detection using Deep Learning models](https://github.com/utsavk28/Nerds/blob/main/notebooks/asl-alphabet-detection-using-dl-models%20(1).ipynb)
* [ASL Alphabet detection](https://github.com/utsavk28/Nerds/blob/main/notebooks/asl-alphabet-s-notebook.ipynb)
* [Sign Language detection](https://github.com/utsavk28/Nerds/blob/main/notebooks/sign-language-detection-2.ipynb)
* [Sign Language detection - 2](https://github.com/utsavk28/Nerds/blob/main/notebooks/sign-language-detection%20(4).ipynb)
* [ASL model testing](https://github.com/utsavk28/Nerds/blob/main/notebooks/asl_recognition_model.ipynb)

## 🛠 Project Setup

1. Clone the repository using the ```git clone```
```
 $ git clone https://github.com/utsavk28/Nerds.git
```
2. Create a virtual environment
```
 $ virtualenv venv
 $ source venv/bin/activate
```
3. Install the required packages
```
 $ pip install -r requirements.txt
```
4. Run the app
```
 $ cd webapp
 $ python app.py
```
## 📸 Results

## 🌐 Conclusion
