# Audio Captcha Recognition

## Introduction

CAPTCHAs are computer generated tests that human can pass but current computer systems can not. They have common usage in various web services in order to be able to detect a human from computer programs autonomously. In this way, owners can protect their web services
from bots. In addition to visual CAPTCHAs which consist of distorted images, mostly test images, that a user must write some description about that image, there are a significant amount of audio CAPTCHAs as well. Briefly, audio CAPTCHAs are sound files which consist of human sound under heavy noise where the speaker pronounces a bunch of digits consecutively. Generally, in those sound files, there are some periodic and non-periodic noises to get difficult to recognize them with a program but not for a human listener. We gathered numerous randomly collected audio file to train and then test them using our SVM algorithm to be able to extract digits out of each conversation.

We used two different measurements the first one is the digit success considering the digits are independent of each other, and the second one is the success of complete recognition of all these digits in the test file where they are usually varying between 4 and 6.

In this research, we used RASTA-PLP features and SVM classifier to recognize audio CAPTCHAs. According to our tests, we reached 98% accuracy for individual digit recognition precision, and around 89% accuracy for the entire digit recognition. We provide a classification algorithm to recognize the digits in the data files and obtain the best accuracy via SVM classifiers using Principle Component Analysis (PCA). In regarding of this goal, we compare our results with non PCA Naive Bayes and default SVM classifier.

## Analysis and Algorithm Techniques

### Data Exploration: 
We provide features from the audio CAPTCHA and use SVM classifier to perform automatic speech recognition on segments of the sample files to elapse the audio CAPTCHAs. There exist several well-known methods for extracting specifications from audio files. The technique that we used here is relative spectral transform-perceptual linear prediction (RASTA-PLP). By using RASTA-PLP, we could be able to train our classifiers to identify words and digits all by itself of who pronounce them.

We can see in Figure 1, a sample audio file is displayed in the time domain and in the frequency domain. In the time domain, the labels of the digits are also shown. 

![Sample image](figures/spect.jpg?raw=true "Title")




In Figure 2, 04648 is pronounced
in the sample file. As you can see, many non-numeric regions resemble those in the time domain
and the frequency domain. There may be periodic and random noisy background speeches that make
automatic identification difficult in non-digitized places.


