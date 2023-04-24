# Handwritten-Digit-Classification
Using LDA, SVM, and Decision Trees to deal with MNIST data set

</p>
Xinqi Chen @23/04/2023 

## Table of Content
- [Handwritten-Digit-Classification](#handwritten-digit-classification)
  - [Abstract](#abstract)
  - [Overview](#overview)
  - [Theoretical Background](#theoretical-background)
  - [Algorithm Implementation and Development](#algorithm-implementation-and-development)
  - [Computational Results](#computational-results)
  - [Summary and Conclusions](#summary-and-conclusions)
  - [Acknowledgement](#acknowledgement)
  
## Abstract
This project involves the classification of handwritten digits using three different classifiers: Linear Discriminant Analysis (LDA), Support Vector Machines (SVM), and Decision Trees. The goal is to compare the performance of these classifiers in terms of accuracy and identify the most difficult and easiest digit pairs to separate. The project uses the popular MNIST dataset, which consists of 70,000 grayscale images of handwritten digits from 0 to 9, each of size 28x28 pixels.
  
## Overview
The project uses Python and popular machine learning libraries such as NumPy, scikit-learn, and Matplotlib to implement and evaluate the classifiers. It follows these main steps:

**Data Preparation**: The MNIST dataset is loaded and preprocessed to create the training and test sets. The images are flattened into 1D arrays, and the pixel values are scaled to the range [0, 1] for better model performance.

**LDA Implementation**: Linear Discriminant Analysis (LDA) is implemented to project the high-dimensional image data into a lower-dimensional subspace. The LDA model is trained on the training data, and the training and test data are transformed into the LDA subspace.

**SVM Implementation**: Support Vector Machines (SVM) with a linear kernel is implemented to classify the digits. The SVM model is trained on the training data, and the training and test data are predicted using the trained model.

**Decision Tree Implementation**: Decision Trees are implemented using the DecisionTreeClassifier from scikit-learn. The Decision Tree model is trained on the training data, and the training and test data are predicted using the trained model.

**Accuracy Calculation**: The accuracy of each classifier is calculated on the test data for each digit pair, and the digit pairs with the highest and lowest accuracy are identified.

**Results and Conclusion**: The results are presented, including the accuracy of each classifier for different digit pairs, and the most difficult and easiest digit pairs to separate are discussed. The conclusion summarizes the findings and provides insights for future work.

## Theoretical Background
### Linear Discriminant Analysis (LDA)
Linear Discriminant Analysis (LDA) is a supervised dimensionality reduction technique that can be used for classification. It seeks to find a projection of the data into a lower-dimensional subspace that maximizes the class separability. In the context of this project, LDA is used to project the high-dimensional image data into a lower-dimensional subspace where the digits can be more easily separated.

### Support Vector Machines (SVM)
Support Vector Machines (SVM) is a popular classification algorithm that can be used for binary and multiclass classification. SVM finds the optimal hyperplane that best separates the data into different classes while maximizing the margin between the classes. In this project, SVM with a linear kernel is used for digit classification.

### Decision Trees
Decision Trees are a type of supervised learning algorithm used for both classification and regression tasks. Decision Trees build a tree-like structure by recursively splitting the data based on the values of features to create decision rules that lead to the correct class labels. In this project, Decision Trees are used for digit classification.

## Algorithm Implementation
The project implements LDA, SVM, and Decision Trees using scikit-learn, a popular machine learning library in Python. The steps involved in each algorithm are as follows:

### LDA
Load the MNIST dataset and preprocess the data. Split the data into training and test sets. 

Perform PCA on the training data
```ruby
U_train, s_train, Vt_train = np.linalg.svd(X_train, full_matrices=False)
Vt_train_pca = Vt_train[:3, :].T  # Use first 3 V-modes for PCA
```

Project the training data onto the first 3 PCA modes. Then, fit an LDA model on the training data.
```ruby
X_train_pca = np.dot(X_train, Vt_train_pca)

# Perform LDA on the projected data
lda = LDA()
lda.fit(X_train_pca, y_binary)
```

Use the model to predict the test data and evaluate the accuracy.
```ruby
# Project the test data onto the first 3 PCA modes
U_test, s_test, Vt_test = np.linalg.svd(X_test, full_matrices=False)
Vt_test_pca = Vt_test[:3, :].T  # Use first 3 V-modes for PCA
X_test_pca = np.dot(X_test, Vt_test_pca)
```


### SVM
Load the MNIST dataset and preprocess the data. Split the data into training and test sets. Then, fit an SVM model on the training data.
```ruby
# Create SVM classifier with default hyperparameters
svm_clf = SVC()

# Fit the SVM classifier to the training data
svm_clf.fit(X_train, y_train)
```
Use the model to predict the test data and evaluate the accuracy.
```ruby
# Predict the labels for the test data
y_pred_svm = svm_clf.predict(X_test)

# Calculate accuracy of SVM classifier
accuracy_svm = accuracy_score(y_test, y_pred_svm)
```

### Decision Trees
Load the MNIST dataset and preprocess the data. Split the data into training and test sets. Then, fit a Decision Tree model on the training data.
```ruby
# Create Decision Tree classifier with default hyperparameters
dt_clf = DecisionTreeClassifier()

# Fit the Decision Tree classifier to the training data
dt_clf.fit(X_train, y_train)
```

Use the model to predict the test data and evaluate the accuracy.
```ruby
# Predict the labels for the test data
y_pred_dt = dt_clf.predict(X_test)

# Calculate accuracy of Decision Tree classifier
accuracy_dt = accuracy_score(y_test, y_pred_dt)
```

## Computational Results

### MNIST data analysis
The first 4 principal components of the data set is plotted as below:
![principal plot](https://github.com/Mmmo-C/Handwritten-Digit-Classification/blob/main/results/modes.png)

The singular value spectrum can be shown as:
![SVS](https://github.com/Mmmo-C/Handwritten-Digit-Classification/blob/main/results/SVS.png)

The interpretation of U, Σ, and V matrices are:
```
Rank of the digit space: 713
Shape of U: (70000, 784)
Shape of Sigma: (784, 784)
Shape of V: (784, 784)
```

The projection of mode 2, 3, and 5 onto 3D space is:
![projection](https://github.com/Mmmo-C/Handwritten-Digit-Classification/blob/main/results/4.png)

### Classification
**LDA**

Firstly, a 2-digit LDA is done. Picking digits 3 and 8 and build a linear classifier, using the first 5,000 data as the training set, and the rest of it as testing set. We can have the accuracy result as:
```
Accuracy: 0.8084923076923077
```

Then, the same LDA is done with 3-digit spectrum. Choosing 1, 3, 8 as the modes, the accuracy we have is:
```
Accuracy: 0.8209076923076923
```

Now, the LDA algorithm is applied to every pair of the digits to find out the most difficult digits to separate and the easiest digits to separate.
```
The most difficult digits to separate are: (0, 1)
Accuracy of separation with LDA on the test data: 0.7854769230769231
The most easy digits to separate are: (1, 2)
Accuracy of separation with LDA on the test data: 0.8813384615384615
```

**SVM**

A SVM classifier is built to perform the same thing as above. 
```
Accuracy of SVM Classifier on test data: 0.9489538461538461
```

Testing the accuracy for each pair of digits:
```
Digit pair with highest accuracy: (0, 1) with accuracy 0.9810016013975833
Digit pair with lowest accuracy: (3, 9) with accuracy 0.9222790023644268
```

**Decision Tree**

A decision tree classifier is built to perform the same classification.
```
Accuracy of Decision Tree Classifier on test data: 0.7718923076923077
```

Testing the accuracy for each pair of digits:
```
Digit pair with highest accuracy: (0, 1) with accuracy 0.8757461056922405
Digit pair with lowest accuracy: (5, 8) with accuracy 0.6837118117954583
```

## Summary and Conclusions
In this project, we explored the performance of different machine learning algorithms for digit classification on the MNIST dataset. We implemented Linear Discriminant Analysis (LDA), Support Vector Machines (SVM), and Decision Trees, and evaluated their accuracy in classifying the digits.

Based on the results, it was found that LDA achieved the highest accuracy in classifying the easiest digit pairs, while SVM performed better in classifying the hardest digit pairs. Decision Trees showed lower accuracy compared to LDA and SVM in most cases.

The computational complexity of the algorithms was also considered, with LDA being the least computationally expensive, followed by SVM, and Decision Trees being the most computationally expensive due to their tree-based structure.

In conclusion, the choice of algorithm for digit classification depends on the specific pair of digits being classified and the trade-off between accuracy and computational complexity. LDA may be preferred for easier digit pairs with lower computational cost, while SVM may be more suitable for harder digit pairs with higher accuracy. Decision Trees could be an option for simpler classification tasks, but may require more computational resources for complex digit pairs.

Overall, this project provides a comparative analysis of three popular machine learning algorithms for digit classification and can serve as a reference for further research and development in this field.

## Acknowledgement
- [ChatGPT](https://platform.openai.com/)
- [Scipy Documentation](https://docs.scipy.org/doc/scipy/)
- [sklearn User Guide](https://scikit-learn.org/stable/user_guide.html#user-guide)





