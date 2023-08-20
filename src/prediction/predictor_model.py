import os
import warnings
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.exceptions import NotFittedError
from sklearn.ensemble import VotingClassifier

warnings.filterwarnings("ignore")


PREDICTOR_FILE_NAME = "predictor.joblib"


class Classifier:
    """A wrapper class for the Ensemble Learner binary classifier.

    This class provides a consistent interface that can be used with other
    classifier models.
    """

    model_name = "Ensemble Classifier"

    def __init__(
        self,

        C: Optional[float] = 1.0,
        degree: Optional[int] = 3,
        coef0: Optional[float] = 0.0,
        tol: Optional[float] = 0.001,
     

        rf_n_estimators: Optional[int] = 40,
        rf_max_depth: Optional[int] = 1,
        min_samples_split: Optional[int] = 2 , 
        min_samples_leaf: Optional[int] = 2 , 
        max_features: Optional[str] = "log2",
        # **kwargs,

        min_child_weight: Optional[int] = 1,
        xg_n_estimators: Optional[int] = 40,
        gamma: Optional[int] = 0,
        
    ):
        """Construct a new Ensemble Learner binary classifier.

        Args:
            min_samples_split (int, optional): The minimum number of samples required
                to split an internal node.
                Defaults to 2.
            min_samples_leaf (int, optional): The minimum number of samples required
                to be at a leaf node.
                Defaults to 1.
        """
        self.rf_n_estimators = rf_n_estimators
        self.max_depth = rf_max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

        self.C = C
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
      

        self.min_child_weight = min_child_weight
        self.xg_n_estimators = xg_n_estimators
        self.gamma = gamma

        self.model = self.build_model()
        self._is_trained = False

    def build_model(self) -> VotingClassifier:
        """Build a new binary classifier."""
        rf = RandomForestClassifier(
            n_estimators=self.rf_n_estimators,
            max_depth=self.max_depth,
            min_samples_split = self.min_samples_split,
            min_samples_leaf = self.min_samples_leaf,
            max_features = self.max_features
        )

        svm = SVC(
            C = self.C,
            degree = self.degree,
            coef0 = self.coef0, 
            tol = self.tol,
            probability=True
        )

        xgboost = XGBClassifier(
            n_estimators = self.xg_n_estimators,
            gamma = self.gamma,
            min_child_weight = self.min_child_weight
        )

        clf = VotingClassifier(estimators = [('svc', svm),('rf', rf),('xgboost', xgboost)],voting='soft') 
        return clf

    def fit(self, train_inputs: pd.DataFrame, train_targets: pd.Series) -> None:
        """Fit the binary classifier to the training data.

        Args:
            train_inputs (pandas.DataFrame): The features of the training data.
            train_targets (pandas.Series): The labels of the training data.
        """
        self.model.fit(train_inputs, train_targets)
        self._is_trained = True

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class labels for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class labels.
        """
        return self.model.predict(inputs)

    def predict_proba(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class probabilities.
        """
        return self.model.predict_proba(inputs)

    def evaluate(self, test_inputs: pd.DataFrame, test_targets: pd.Series) -> float:
        """Evaluate the binary classifier and return the accuracy.

        Args:
            test_inputs (pandas.DataFrame): The features of the test data.
            test_targets (pandas.Series): The labels of the test data.
        Returns:
            float: The accuracy of the binary classifier.
        """
        if self.model is not None:
            return self.model.score(test_inputs, test_targets)
        raise NotFittedError("Model is not fitted yet.")

    def save(self, model_dir_path: str) -> None:
        """Save the binary classifier to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Classifier":
        """Load the binary classifier from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Classifier: A new instance of the loaded binary classifier.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return (
            f"Model name: {self.model_name} ("
            f"min_samples_leaf: {self.min_samples_leaf}, "
            f"min_samples_split: {self.min_samples_split})"
        )
    

def train_predictor_model(
    train_inputs: pd.DataFrame, train_targets: pd.Series, hyperparameters: dict
) -> Classifier:
    """
    Instantiate and train the predictor model.

    Args:
        train_X (pd.DataFrame): The training data inputs.
        train_y (pd.Series): The training data labels.
        hyperparameters (dict): Hyperparameters for the classifier.

    Returns:
        'Classifier': The classifier model
    """
    classifier = Classifier(**hyperparameters)
    classifier.fit(train_inputs=train_inputs, train_targets=train_targets)
    return classifier

def predict_with_model(
    classifier: Classifier, data: pd.DataFrame, return_probs=False
) -> np.ndarray:
    """
    Predict class probabilities for the given data.

    Args:
        classifier (Classifier): The classifier model.
        data (pd.DataFrame): The input data.
        return_probs (bool): Whether to return class probabilities or labels.
            Defaults to True.

    Returns:
        np.ndarray: The predicted classes or class probabilities.
    """
    if return_probs:
        return classifier.predict_proba(data)
    return classifier.predict(data)


def save_predictor_model(model: Classifier, predictor_dir_path: str) -> None:
    """
    Save the classifier model to disk.

    Args:
        model (Classifier): The classifier model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Classifier:
    """
    Load the classifier model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Classifier: A new instance of the loaded classifier model.
    """
    return Classifier.load(predictor_dir_path)

def evaluate_predictor_model(
    model: Classifier, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the classifier model and return the accuracy.

    Args:
        model (Classifier): The classifier model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the classifier model.
    """
    return model.evaluate(x_test, y_test)