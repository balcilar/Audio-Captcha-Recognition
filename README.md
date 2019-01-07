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

In Figure 2, 04648 is pronounced in the sample file. As you can see, many non-numeric regions resemble those in the time domain
and the frequency domain. There may be periodic and random noisy background speeches that make automatic identification difficult in non-digitized places.

![Sample image](figures/timedomain.jpg?raw=true "Title")


### Algorithms and Techniques 
The Matlab functions for extracting features were published and distributed worldwide at and as part of the Voicebox package. We utilize SVM algorithms to carry out automatic digit recognition. We will explain our performance in detail. The benefits of using SVM library on Python Scikit-learn is that they allow the immediate training of high amount of linear classifiers. Another advantage of SVMs is that they can be kernelled to solve non-linear classification tasks conveniently. The main idea of kernel method to handle linearly 
inseparable data is to build non-linear combinations of the basic features to project them on to a higher dimensional space via mapping a function ϕ that will be linearly separable. To solve SVM problems, we need to transform our training set on to a higher dimensional feature space using the function ϕ and train a linear SVM model to classify the data in our new feature space. Therefore, we may use that same ϕ to transform the unknown data to classify it by linear SVM model.

For our problem, we divide the files of the training set into 11 separate classes as 0, 1, 2, ... , 9 and the noise class. We mentioned in Section 3 how many data points we have from each class. Then we extracted the features of these audio signals by using RASTA-PLP. As a feature of RASTA-PLP we obtained 13×42 dimensional feature space. In this number of dimension, 13 is the number of RASTAPLP coefficients and 42 is the number of sliding windows that is, every digit have been scanned by 42 sliding windows, and then 13 coefficients have been calculated from each sliding window. Once we calculated the RASTA-PLP coefficients of the digits where they belong to the generated 11 classes, we apply PCA to reduce dimensionality, and then we made a classification using multiclass SVMs.

Henceforth, an existing test audio file has been fragmentized into sliding windows as we did for the training files before. Then, it has been extracted 546 RASTA-PLP features for each sliding window and we used PCA to reduce the dimensionality and SVM to classify these reduced number of features. If the output of the classification results a digit from 0 to 9, then we assigned it as an element of the
digit class. Otherwise, if the output results a noise or an awkward silence, then we assigned it as an element of the so-called 11th class.

### Benchmark Model 

Without using PCA, we consider naive bayes method and SVM as benchmark models to compare our proposed model. Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem with the naive assumption of independence between every pair of features. These classifiers have worked well in many problems, such as text classification and spam filtering. They do not require a large amount of training data to estimate the involved parameters. Naive bayes method can be faster than SVMs when you compare them. On the other hand, although they are know as a smooth classifiers we can not say they are good estimator. Thus, we applied PCA to our SVM algorithm.

## The Methodology of our Model

### Data Preprocessing 

In the proposed algorithm, preprocessing is used to determine the phonemes that are likely to be digitized. Therefore, there is no need for a preprocessing step in the train set, since the starting points for each digit set plus the non-digit set of 11th are manually set. When extracting the features of the test set, it is necessary to automatically determine the regions that are likely to be digitized. Because now nobody has the possibility to run these zones manually. We implemented a pre-processing algorithm for this. During this preprocessing phase, the test audio signal is read, 0 is averaged, the signal’s energy is calculated, and then the 100-point mean running average of the energy signal is calculated. The smoothed energy sequence is then run to determine the start of the potential
digit regions with some hard thresholding operation. For this, the smoothed energy sequence is called cluster 1 for large parts from 0 to some degree (larger than 0.00001), and cluster 2 for parts smaller than 0.001. Then all the start-end point pairs are calculated for the potential digit blanks that start with the element of cluster 1 and end with the element of cluster 2.

### Implementation & Refinement  
We tested the accuracy of audio CAPTCHAs used by popular machine learning techniques algorithmically planned to break them. Two different measurements were used for the accuracy of the classification. First, we considered the digits independently to obtain accurate digit prediction. Secondly, we measured the prediction accuracy of the digits where they are usually varying between 4 and 6 for the test files. We operated DTW algorithm to perform the difference between the ground truth and predicted digits.
For feature extraction we use RASTA-PLP speech analysis by applying the following steps:
• Calculate the crucial-band spectrum (as in the PLP) and take its log.
• Approximate the temporal derivative of log crucial-band spectrum using 4 consecutive spectral
values.
• Apply SVM as nonlinear classifier for threshold filtering.
• Integrate log crucial-band temporal derivative.
• According to ordinary PLP, add equal noise and multiply by 0.33 to create the power law of
hearing
• Operate exponential function of this log spectrum to produce audio spectrum.











