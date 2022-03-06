# GesturalAI

<div align="center" >
 <img src='https://github.com/utsavk28/GesturalAI-Nerds/blob/main/images/gesturalai-demo.gif' />
 </div>
 
[Full Demo](https://www.youtube.com/watch?v=F3SxnL08jgM)
<!-- #### Note: This is a rough Readme , this Readme will be updated soon with all the details -->

## 📌 About
Sign Language Recognition using Machine and Deep Learning 

## 🎯 Key Features
* Classifies the American Sign Language to english letters in real time by taking video input using webcam

## 💻 Development Process
### Various Techniques/Models Used

* **I3D Transfer Learning** : We worked on classifying most common words in English from sign language. For this, we used the WLASL dataset and fine tuned the I3D model which was pretrained on the Kinetics-400 dataset
* **CNN + RNN with self made dataset** : We Created our own small dataset for 10 common words. Each word had 30 videos of 30 frames. Here we extracted the landmarks of pose, left hand and right hand using mediapipe holistic model. And used those landmarks of the frames for training the RNN model. This Model works accurately for the person who trained but the accuracy drops for the other persons 
* **Transfer Learning with Resnet & Mobilnet** : Trained on ASL dataset (consisting of sign languages of A-Z letters and some special characters like SPACE, DELETE and NOTHING). Predicts accurately on the training, validation and testing dataset but the accuracy drops in production
* **MediaPipe Feature Extraction + Machine Learning Algorithms** : Using MediaPipe's Hand Model , features are extracted and engineered. Those are fed to machine learning model and are used to predict the labels

### Datasets used
* [WLASL video dataset](https://dxli94.github.io/WLASL/)
  * WLASL is the largest video dataset for Word-Level American Sign Language (ASL) recognition
  * This dataset consists of around 20K videos for sign language representation of around 2000 commonly used words

* [American Sign Language Dataset for letters](https://www.kaggle.com/grassknoted/asl-alphabet)
  * Image data set for alphabets in the American Sign Language 
  * Contains 87000 images of 200 X 200 pixels, divided into 29 classes (A - Z, SPACE, DELETE and NOTHING)
  * The test data set contains only 29 images, 1 image per labels.

* [American Sign Language Dataset for letters](https://www.kaggle.com/kapillondhe/american-sign-language)
  * RGB image dataset of American sign language alphabets.
  * This dataset contains 166k images of 200 X 200 pixels, divided into 28 classes (A - Z, SPACE and NOTHING)
  * The test data set contains 112 images , 4 images per labels.

### Notebooks
[Notebook's Folder Link](https://drive.google.com/drive/folders/1j_iCenuO-dUMK05K94yNbB9xmKwQR7uH?usp=sharing)

* [Action Recognition](https://drive.google.com/file/d/1yEKdP3v4eF2CtTqX_Mw415k8gr9aLRIn/view?usp=sharing)
    * Hand Feature Extraction on `WLASL` Dataset & feeding the output to RNN for Word Sign Classificaion 
* [ASL Alphabet detection using Deep Learning models](https://drive.google.com/file/d/1bWY6CGYOwnz6S6aCV4UH_3pG0A1lgKPi/view?usp=sharing)
* [ASL Alphabet detection](https://drive.google.com/file/d/1xmgb05Y98F1qQiFQOwIccRJjDW4YY1M0/view?usp=sharing)
* [Sign Language detection](https://drive.google.com/file/d/1DN_S6dHHx2lsg1j7EI9h0GnmfrpR-OU1/view?usp=sharing)
* [Sign Language detection - 2](https://drive.google.com/file/d/1TB7UYQeqnakcA7KsDm6sjJQYeUAUCPOT/view?usp=sharing)
* [ASL model testing](https://drive.google.com/file/d/10Y1nfoVWflWSCzVoW5mQPA97uJE1clOr/view?usp=sharing)

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
 $ python main.py
```

## Group members
- [Utsav Khatu](https://github.com/utsavk28)
- [Tushar Bauskar](https://github.com/tusharsb-12)

## ASL characters and their sign representations
![ASL characters](https://github.com/utsavk28/Nerds/blob/main/images/ASL%20characters.png?raw=true)

## 📸 Results
[Demo link](https://www.youtube.com/watch?v=F3SxnL08jgM)
<!-- 
## 🌐 Conclusion -->
