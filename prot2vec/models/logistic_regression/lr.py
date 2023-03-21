import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from rich.progress import track
from joblib import dump, load, Parallel, delayed
from typing import List, Tuple, Union
from pathlib import Path


import os
import logging

log = logging.getLogger(__name__)

class GOTermLogisticRegression:
    
    def __init__(self, model_dir:Union[str,Path]) -> None:
        self.trained_ = False
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=False, exist_ok=True)
        self.models_ = {}

    def trained_model_exists_(self, term_name:str) -> bool:
        goterm = term_name.replace(":", "_")
        model_filename = self.model_dir / goterm
        return model_filename.exists()

    def save_trained_model_(self, term_name:str, model):
        goterm = term_name.replace(":", "_")
        dump(model, self.model_dir / goterm)

    def load_trained_model(self):
        log.info(f"Loading pre-trained models located in {self.model_dir}")
        models = list(self.model_dir.iterdir())
        n = len(models)
        for saved_model in track(models, description="Loading...", total=n):
            term_name = saved_model.name.replace("_", ":")
            self.models_[term_name] = load(saved_model)

    def fit_single_(self, X, y, term_name:str) -> None:
        lr_model = SGDClassifier(loss='log', n_jobs=35)
        lr_model.fit(X, y)
        self.save_trained_model_(term_name, lr_model)

    def fit(self, X, y, terms_index: List[str]):
        """
        Fit classifier with training data

        Parameters
        ----------
        X : numpy.ndarray
            input features of size
            :code: `(n_samples, n_features)`
        y : numpy.ndarray
            binary indicator matrix with label assignments
        terms_index : list of str

        Return
        ------
        self
            fitted instance of self
        """
        log.info(f"Detecting already trained models...")
        untrained = [(i, term) for i, term in enumerate(terms_index) if not self.trained_model_exists_(term)]
        log.info(f"Training Logistic regression for {len(untrained)} GO terms")
        #Parallel(n_jobs=8, max_nbytes=1e6)(
        #    delayed(self.fit_single_)(X, y[:, i], term) for i, term in untrained
        #)
        for i, term in track(untrained, description="Training..."):
            self.fit_single_(X, y[:, i], term)

        self.trained_ = True

    def predict(self, X, terms_index: List[str]):
        assert self.trained_, log.error("Can't predict using an untrained model") 
        pass







