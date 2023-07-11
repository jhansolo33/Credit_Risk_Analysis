# Credit_Risk_Analysis

##  Overview
In 2019, more than 19 million Americans had at least one unsecured personal loan. Personal lending is growing faster than credit card, auto, mortgage, and even student debt. With that type of growth, FinTech firms are growing muvh faster than traditional loan processes. By using the latest machine learning techniques, FinTech firms are able to continuously analyze large amounts of data and predict trends to optimize lending risk. In this project we utilize Python to build and evaluate several machine learning models to predict credit risk learning algorithms which can help banks and financial institutions predict anomalies, reduce risk cases, monitor portfolios, and provide recommendations on what to do in cases of fraud. Skills that were learned:

• Explain how a machine learning algorithm is used in data analytics.

• Create training and test groups from a given data set

• Implement the logistic regression, decision tree, random forest, and support vector machine algorithms

• Interpret the results of the logistic regression, decision tree, random forest, and support vector machine algorithms

• Compare the advantages and disadvantages of each supervised learning algorithm

• Determine which supervised learning algorithm is best used for a given data set or scenario

• Use ensemble and resampling techniques to improve model performance.

##  Analysis Reports Used to Predict Credit Risk
1.  Oversampling Models: naive random oversampling algorithm and the SMOTE algorithm
2.  Undersample Model: Cluster Centroids algorithm
3.  Resample Model: used the SMOTEENN algorithm SMOTEENN Algorithm
4.  Classification Report assessed the performance of two ensemble algorithms; training a Balanced Random Forest Classifier and an Easy Ensemble AdaBoost classifier; each algorithm Ensemble Classifiers generate the imbalanced_classification_report from imbalanced-learn.
5  
##  Resources
###  Data Sources: Module(18)-Challenge-Resources.zip and LoanStats_2019Q1.csv

###  Data Tools: credit_risk_resampling_starter_code.ipynb and credit_risk_ensemble_starter_code.ipynb.

###  Software: Python, Visual Studio Code, Anaconda, Jupyter Notebook, and Pandas

###  Python Ecosystem libraries:
NumPy
SciPy
Scikit-learn
imbalanced-learn package in the nlenv environment

##  Results: Resampling Models to Predict Credit Risk
###  Oversampling

![Image 1](https://github.com/jhansolo33/Credit_Risk_Analysis/assets/119264589/5de7bfcd-ccb3-4e8f-8c0d-bd4ab72f0d7d)
![image 2](https://github.com/jhansolo33/Credit_Risk_Analysis/assets/119264589/f32895ac-83db-4d5a-a1fb-302b0e7121de)



###  Undersampling
Also, testing an undersampling algorithms to determine which algorithm results in the best performance compared to the oversampling algorithms above. The undersampling of the data done by the Cluster Centroids algorithm.

![Image 3](https://github.com/jhansolo33/Credit_Risk_Analysis/assets/119264589/bfc3a8f8-b343-41ed-8f48-2dd271897016)

###  Over/Under Sampling: SMOTEENN
Another test combined over- and under-sampling algorithm to determine if the algorithm results in the best performance compared to the other sampling algorithms above.

![image 4](https://github.com/jhansolo33/Credit_Risk_Analysis/assets/119264589/28da3f15-9da7-46b6-88ca-ddeb3df95826)


###  Ensemble Classifiers
We compared two ensemble algorithms to determine which algorithm results in the best performance. You will train a Balanced Random Forest Classifier and an Easy Ensemble AdaBoost classifier.

##Balanced Random Forest Classifier:

Predicted High_Risk	Predicted Low_Risk
Actual High-Risk	65	36
Actual Low-Risk	1692	15412
                   pre       rec       spe        f1       geo       iba       sup

  high_risk       0.04      0.64      0.90      0.07      0.76      0.56       101
   low_risk       1.00      0.90      0.64      0.95      0.76      0.59     17104

avg / total       0.99      0.90      0.65      0.94      0.76      0.59     17205

##Easy Ensemble AdaBoost Classifier

	Predicted 0	Predicted 1
  Actual 0	65	36
  Actual 1	1692	15412
              Classification Report
              precision    recall  f1-score   support

   high_risk       0.04      0.64      0.07       101
    low_risk       1.00      0.90      0.95     17104

    accuracy                           0.90     17205
   macro avg       0.52      0.77      0.51     17205
weighted avg       0.99      0.90      0.94     17205



###  Terms
The ML models process of fitting, reshaping, and training the same data is carried out significantly different. The evaluating parameters followed by a description of their origin follow:

ACCURACY SCORE reports a percentage of precision of the predictions compared to the actual results. However, it is not enough just to see that results, especially with unbalanced data. Equation: accuracy score = number of correct prediction/total number of predictions.

PRECISION is the measure of how reliable a positive classification is. A low precision is indicative of a large number of false positives. Equation: Precision = TP/(TP + FP)

RECALL is the ability of the classifier to find all the positive samples. A low recall is indicative of a large number of false negatives Equation: Recall = TP/(TP+FN)

F1 SCORE is weighted average of the true positive rate (recall) and precision, where the best score is 1.0. Equation: F1 score = 2(Precision x Sensititivity)/(Precision + Sensitivity) The F1 Score equation is: 2*((precisionrecall)/(precision+recall)). It is also called the F Score or the F Measure. Put another way, the F1 score conveys the balance between the precision and the recall. The F1 for the All No Recurrence model is 2((0*0)/0+0) or 0.

##  Analysis
Based on the accuracy scores, the Ensemble Classifiers proved to be the most precise. EasyEnsembleClassifierp provides a highest Score for all Risk loans. The precision is low or none for all the models. In general, above the 90% of the current analysis, utlizing EasyEnsembleClassifier will perform a High-Risk loan precision as a great value for the overall analysis. Models, Naïve Random Over Sample, SMOTE Oversampling, Cluster Centroids Undersampling and SMOTEENN.

A well know principle, “Accuracy matters” yet only up to a certain extent. Other performance metrics like Confusion Matrix, Precision-Recall, and F1-Score should be consider along with Accuracy while evaluating a Machine Learning model. Precision for all four models are 0.01 for high-risk loans and 1.00 for low risk loans. Low precision scores for high-risk loans is based on the large number of false positives, meaning that many of low-risk loans were marked as high-risk loans. High score for low-risk loans indicate that nearly all low risk scores were marked correctly; however, lower recall score (0.58 for naive Naive Random Oversampling and Logistic Regression, for example) indicates that there were quite a few low risks loans that were market as high risk, when that was not the case. Actual high-risk loans have slightly better scores on recall (0.77 for naive Naive Random Oversampling) indicating that there weren't as many false negatives or not many high-risk loans were marked as low risk loans.

Generally speaking, the precision-recall values keep changing as you increase or decrease the threshold. Building a model with higher precision or recall depends on the problem statement you’re dealing with and its requirements.

Precision-Recall values can be very useful to understand the performance of a specific algorithm and also helps in producing results based on the requirements. But when it comes to comparing several algorithms trained on the same data, it becomes difficult to understand which algorithm suits the data better solely based on the Precision-Recall values. The F1 score characterized as a single summary statistic of precision and sensitivity. For the ensemble the high-risk 0.07 and low-risk is 0.95 occurs when the sensitivity is very high, while the precision is very low. We have a trade-off between sensitivity and precision, and that a balance must be struck between the two. A useful way to think about the F1 score is that a pronounced imbalance between sensitivity and precision will yield a low F1 score. Having a precision or recall value as 0 is not desirable and hence it will give us the F1 score of 0 (lowest). On the other hand, if both the precision and recall value is 1, it us the F1 score of 1 indicating perfect precision-recall values. All the other intermediate values of the F1 score ranges between 0 and  1.

The Ensemble model’s accuracy scores are both 90% for Balanced Random Forest Classifier and Easy Ensemble AdaBoost Classifier. Recall scores for both model and both - low and high-risk scores and precision for low risk were high, meaning very good accuracy. Precision for high-risk loans in both models were not high at 0.07 for both Balanced Random Forest Classifier and Easy Ensemble AdaBoost Classifier indicating that there were large number of false positives, meaning that large number of low-risk loans were marked as high risk.

The ensemble models demonstrated more accuracy than the other four models. However, they might be prone to overfitting. If that occurs and we don't get desired results when working with new data set, we can do some further fine-tuning (pruning) to avoid the overfitting. Observing the confusion matrix along with the accuracy scores, an identification that the model is overfitting on the training dataset could be made as it is predicting every unknown data point as a low-risk loan. If it wasn’t for the confusion matrix, we would have never known the underlying issue. Using these metrics, the Confusion Matrix, Precision-Recall, and F1 Score, assisted in refining the evaluation of the model’s performance. Suggestions for the future would be to evaluate using a different data set or using other machine learning algorithms.
