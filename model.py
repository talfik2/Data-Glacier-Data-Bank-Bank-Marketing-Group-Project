
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
