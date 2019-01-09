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
  * Calculate the crucial-band spectrum (as in the PLP) and take its log.
  * Approximate the temporal derivative of log crucial-band spectrum using 4 consecutive spectral
values.
  * Apply SVM as nonlinear classifier for threshold filtering.
  * Integrate log crucial-band temporal derivative.
  * According to ordinary PLP, add equal noise and multiply by 0.33 to create the power law of
hearing
  * Operate exponential function of this log spectrum to produce audio spectrum.
  
To reduce the dimensionality, we used PCA since redundancy were growing as the dimension of the data increased. PCA is one of the most common method to utilize dimensions by transforming them to a new space. We applied k-fold cross validation (in this case k = 4) to decide the dimension for the usage of the classifier and found the optimum penalty parameter C for the SVM algorithm. The variance parameter used by the PCA algorithm and the k-fold cross validation of the train set were used to determine the best value of the penalty parameter that the SVM algorithm would need. When subtracting the features of the train set, a signal of 0.4 seconds is taken from each digit, plus the manually set start point for the non-digits of the eleventh set. It is assumed here that each digit is read and finished in 0.4 seconds, which is quite sufficient. This signal is divided into 42 sliding subwindows that overlap 50% and 13  random-plp coefficients of each sub-window are calculated. Then these coefficients are combined for 42 sub windows to obtain 42 × 13 = 546 feature vectors. Now an audio domain, labeled as one of 11 classes, is now expressed in 546 numbers, so its features have been
extracted.

Before we extract the features of a test element, the data pre-processing determines the potential start point of the audio block. Then, from each starting point, another 0.4 second segment is taken and divided into 42 sliding windows overlapping 50% in the same way as train data. 13 random-plp coefficients of each sub-window are calculated. Then, these coefficients are combined for 42 subwindows to obtain 42 × 13 = 546 feature vectors. We now have a 546-length feature vector of this potential phoneme. This 546-length vector is classed according to the method to be applied and is determined to be one of 11th class according to the output of the method. Refinement executed by conducting cross validation so as to determine the most optimum values of PCA variance and C-value.

### Model Evaluation & Validation 

The 4-fold cross validation was applied to the train set to determine the best value of the penalty parameter that the SVM algorithm would need with the variance variable used by the PCA algorithm. According to our algorithm, the entire train set is divided into 4 parts randomly, one part is tested and the remaining 3 parts are accepted as train. The system is trained with the given parameters, then the digit performance for the set determined. In this study, we used 11 different penalty parameters of 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, and 7 different penalty values of 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99 where for each penalty value we used the
value of the variance representation. As a result, 4-fold validation was performed separately for 77 different parameters and the validation success was calculated for each case. The following table shows that the calculated validation accuracies for the varying penalty parameter and PCA variance. As you can see from there, the highest validation success was calculated for Penalty Parameter = 50, PCA-Var = 0.9.

![Sample image](figures/crossval.jpg?raw=true "Title")


## Conclusion

The Na¨ıve Bayes method correctly identifies about 42% of the test digits. This method also failed because every class element in the train set is not balanced because the train set has a large number of noise class (11th class) elements, while the elements from 0 to 9 are somewhat less. None of the 100 audio files in the test set were fully recognized by Na¨ıve Bayes method. Since the number of elements is not balanced, even though the classes from 0 to 9 a are partially recognized even in the train set, the noise class has a relatively low success rate of 71%. This is why all test audio files are not recognized without error, yet even if a section is wrongly assigned to the noise class, it means that the test element is misclassified.

On the other hand, when the default number of SVM elements is unstable, it has succeeded in achieving 92% independent digits, but only 44% of the test elements have been achieved. With the parameter we have optimized, the recognition of the complex chaptche sound file has reached 89%. Independent digit success has reached a very good value of 98%. We applied 3 different algorithms naive bayes (no PCA), default SVM (no PCA) and SVM (with PCA) to optimize the parameters as promised. The naive bayes method failed because each class element in the train set was not balanced. Because there are a large number of noise classes in the train cluster, the elements from 0 to 9 are somewhat less. None of the 100 audio files in the test set were fully recognized by Na¨ıve Bayes. Once we compare the results of our algorithm for each case we found the following distribution in following table.

![Sample image](figures/results.jpg?raw=true "Title")


## How to Run

After clone our repository, extract already extracted feature matrix named train_features.zip file. After extraction you can find train_features.mat file which consist of thousand of audio's rasta-plp features. Then you can train the model by given penalty parameter and PCA variance such as;

```
$ python src/captcha_train.py dataset/train/train_features.mat 50 0.9
```

within that command you can find trained model named "train_features_50.0_0.9" in the current folder. After you can test model by all provided test audio files by following command

```
$ python src/captcha_test.py  dataset/train/train_features_50.0_0.9  --all
```

## How to Cite

If you use this code or the publicly shared model, please cite the following paper.

Ahmet Faruk Çakmak and Muhammet Balcilar, "Audio Captcha Recognition Using RastaPLP Features by SVM", 2019. (arXiv)
https://arxiv.org/abs/1901.02153

