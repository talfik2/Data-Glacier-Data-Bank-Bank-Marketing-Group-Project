"""
Steps Covered in this file
1. Importing Libraries
2. Creating df and calling the data
3. Train Test Split
# 4. Defining Model(No import is possible for this specific model)
5. Creating pipeline to implement the model(depending on the model and scenario, pipeline may not be needed for all use cases
6. Training and testing the model(in this case, pipeline)
7. Saving the model to Joblib(That's how the joblib works :))
- Joblib is used for saving and loading ML Models
8. Calling the model from Joblib
"""

# Save Model Using Pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


banking = "C:\\Users\\talfi\\python\\dataglacier\\w9\\banking.csv"

dataframe = pd.read_csv(banking)
# y = pd.read_csv(Y)

# Splitting data into train & test sets
X_train, X_test, y_train ,y_test = train_test_split(
    banking.drop(columns = "y"),
    banking["Company"],
    test_size=0.25,
    random_state=42,
    stratify=banking["y"]
)

# Defining stacking estimator

from sklearn.base import BaseEstimator, TransformerMixin, is_classifier
from sklearn.utils import check_array
class StackingEstimator(BaseEstimator, TransformerMixin):

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):

        self.estimator.fit(X, y, **fit_params)
        return self

    def transform(self, X):

        X = check_array(X)
        X_transformed = np.copy(X)
        # add class probabilities as a synthetic feature
        if is_classifier(self.estimator) and hasattr(self.estimator, 'predict_proba'):
            y_pred_proba = self.estimator.predict_proba(X)
            # check all values that should be not infinity or not NAN
            if np.all(np.isfinite(y_pred_proba)):
                X_transformed = np.hstack((y_pred_proba, X))

        # add class prediction as a synthetic feature
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))
        return X_transformed

# Creating our pipeline

from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
st = StackingEstimator(estimator=DecisionTreeClassifier(max_depth=10,
                                                   min_samples_leaf=11,
                                                   min_samples_split=4,
                                                   random_state=42))
gnb = GaussianNB()

pipeline = make_pipeline(st, gnb)

# Fit the model on training set
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)


# save the model to disk with joblib
filename = 'model.pkl'
joblib.dump(pipeline, 'model.pkl')

 
# some time later...

#load the model from disk with joblib
pipeline = joblib.load('model.pkl')
