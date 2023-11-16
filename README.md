# ML-Intro
Compilation of different Machine Learning algorithms implemented in Python during the practices of the subject Machine Learning I, second year.
All practices have been conducted in Spanish and their corresponding conclusions are in Spanish.

## 1. Linear Regression
Linear regression is a basic predictive model used in machine learning. The practice includes information on how to apply it in machine learning projects, how to use the Matplotlib library to visualise the data and how to use the NumPy library to perform numerical calculations. Also included are practical examples of how to apply linear regression to a dataset of runner times from an ultra-marathon race.

## 2. Linear Regression II
The practice consists of dividing a dataset into a training and a test set, and then training 8 linear regression models without regularisation using different combinations of features. For each model, the RMSE value over the test set is obtained and presented in tabular form.

## 3. Perceptron I
Application of the Perceptron algorithm for data classification in two different contexts. In the first part, the construction of decision surfaces using sklearn's Perceptron class is explored, using datasets extracted from Iris flowers to visualise and understand its ability to separate classes. In the second part, synthetic data is generated and the Perceptron's ability to classify synthetic samples through its decision function is evaluated. These exercises not only allow us to become familiar with the Perceptron's performance in linear classification, but also to analyse its results and understand its usefulness in different data classification contexts.

## 4. Perceptron II
Implementation of the Perceptron algorithm for binary classification problems. A Perceptron class will be created with arguments including the maximum number of iterations, the ability to shuffle samples after each epoch and the fraction value to update the vector of weights. This class will contain attributes such as 'weights_' and 'threshold_weights_'. Methods such as 'fit' to train the classifier using the Perceptron algorithm and 'predict' to obtain the classes corresponding to a set of samples will be implemented. In Exercise 2, exercise 1 will be repeated using this implemented Perceptron instead of the Perceptron class provided by sklearn. This task allows for a direct understanding and application of the Perceptron algorithm, facilitating comparison with the sklearn implementation and promoting a deeper understanding of how it works in data classification.

## 5. Logistic Regression
In this exercise, different datasets are used for two specific exercises. In Exercise 1, synthetic data is generated with the make_blobs function, creating a dataset with two features and three classes, whose centres are at predefined coordinates. 20 samples per class are generated for the creation of this dataset. In Exercise 2, a subset of 10,000 faces from the CelebA dataset is used, containing images of famous people along with 40 annotated attributes, such as gender, hair colour and smile, among others. Each image has an associated 512-feature vector obtained using the ArcFace network. The training and test sets are contained in CelebA-10K-train.csv and CelebA-10K-test.csv files, respectively. They will be used to train a two-class linear classifier using Logistic Regression and measure the hit rate on the test samples. The optional exercise involves classifying additional images provided in a compressed file (ImagesToClassify.zip) using the classifier and flagging those that are incorrectly classified. These exercises allow comparison of the effectiveness of different classification methods on synthetic and real data sets, encouraging understanding of the algorithms and evaluation of their performance in practical situations.

## 6. Nearest neighbours
Exercise 1 focuses on classifying blocks of data using the nearest neighbour method. The KNeighborsClassifier class from the sklearn library is used for this purpose. We divide the data into training and test sets, evaluate the accuracy by varying the number of neighbours and repeat this process after scaling the features. We analyse the runtime for different search algorithms and determine the best combination of parameters, comparing results with and without scaling.

In Exercise 2, a nearest centroid classifier based on Algorithm 7.1 is implemented using the NearestCentroid class. Its performance is compared with sklearn's NearestCentroid class on a given dataset. The results are analysed using the confusion matrix to understand and compare the effectiveness of both implementations in data classification.

## 7. Vector Machines Support
Exercise 1 focuses on classifying a synthetic data set of three classes in a two-dimensional space. A Linear Support Vector Machine (SVM) is used to obtain optimal decision surfaces that maximise the separation between samples of different classes. The SVC class of the sklearn svm module is used, varying the regularisation values (C = {0.01, 1, 10}) and visualising the training samples, decision surfaces and support vectors graphically.

In Exercise 2, datasets derived from celebrity facial images are used. It is proposed to implement a gender classifier using SVM with different kernels (linear, polynomial and radial) and C regularisation values. The ensemble is divided into training and validation to find the best combination of parameters, evaluating the hit rate on the test set. Additional image classification is offered as an option to compare the results with previous practice. These exercises explore the application of SVM in synthetic and real-world problems, analysing how different configurations influence classification accuracy.

## 8. Bayesian classifier
In Task 1, the Naive Bayes classifier is applied to message datasets, dividing them into training and testing to classify messages as spam or legitimate. Several Naive Bayes implementations available in the sklearn library are used, such as GaussianNB, MultinomialNB, ComplementNB and BernoulliNB. Their performance is evaluated through accuracy and confusion matrices are obtained to analyse the results and compare the performance of each classifier.

In Task 2, the performance of the Naive Bayes classifier is compared with other known classifiers, such as Perceptron, Logistic Regression, SVM and Nearest Neighbours, in classifying spam messages. The same datasets are used and the hit rate of each classifier is evaluated to determine how they compare in identifying legitimate and spam messages. A thorough analysis of the results is performed to understand the strengths and weaknesses of each model in this classification task.

## 9. Performance estimation I
In Exercise 1, we work with the confusion matrix to evaluate the performance of five different classifiers trained on a specific dataset. The steps are:
1. Train five different classifiers using data from the aforementioned dataset.
2. Generate and graphically display confusion matrices for each classifier.
3. Calculate performance measures such as accuracy, recall, precision, specificity and F1-measure using the confusion matrix values for each classifier. These values will be presented in tabular form to compare the performance of each classifier against these metrics.

In Exercise 2, the performance of the same five classifiers in ROC space is explored. The steps are:
1. Retrain the five classifiers using the aforementioned data set.
2. Obtain the confusion matrices and, from their values, calculate the True Positive Rate (TPR) and False Positive Rate (FPR) for each classifier.
3. Graphically represent the performance of the classifiers in the ROC space using TPR vs. FPR.
4. Analyse and understand the results obtained, evaluating how each classifier performs in terms of true and false positives.

In Exercise 3, the probability of class membership is used to plot ROC curves for two specific classifiers, a Bayesian classifier, a logistic regression classifier and a support vector machine. The steps are:
1. Train the three classifiers with the data provided in section 10.2.
2. Use the predict_proba method to obtain the probabilities of belonging to each class for each classifier.
3. Plot the ROC curves of the three classifiers on a single graph.
4. Analyse the results obtained, comparing the ROC curves of the different classifiers to assess their performance in terms of sensitivity and specificity.

These exercises provide a detailed view of the performance of the classifiers in terms of different metrics and performance curves, allowing a comprehensive comparison of their effectiveness in classifying the specific dataset.

## 10. Decision Trees
In the first exercise, the sklearn library is used to create and evaluate decision trees for cell classification. We split the data into training and test sets, train a tree with default parameters, search for the best combination of parameters considering the medical nature of the problem and compare the resulting trees graphically.

In the second exercise, feature selection is explored using decision trees. A tree is trained and the features used are identified, then this feature set is used to train a Support Vector Machine (SVM). Subsequently, an equivalent set of features is selected using SelectKBest and another SVM is trained. The performance of both SVMs is compared using the different feature selections obtained. These exercises focus on the use of decision trees for both classification and feature selection in a medical context.

## 11. Performance estimation II
In the first exercise, we compare the performance of linear regression models and regression trees to estimate the value of houses. The repeated holdout technique is used to evaluate both models on different node splitting criteria for the trees and the results are compared using metrics such as mean value and standard deviation.

In the second exercise, you are asked to implement a function called validation_cv that will perform cross-validation. This function should take a dataset, the desired number of partitions, optionally allowing to shuffle the data and set a seed for reproducibility. The function returns a list of the resulting partitions in the form of tuples with indices for training and test set.

# Conclusion
This document summarizes various practical exercises from a Machine Learning I course, covering a range of algorithms and techniques in Python. Each exercise tackles specific topics, from basic linear regression and perceptron algorithms to advanced concepts like SVMs, decision trees, and Bayesian classifiers. Here's a comprehensive summary:
