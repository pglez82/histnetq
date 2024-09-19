from abc import ABCMeta, abstractmethod
import six
import warnings
import math
import numpy as np
from scipy.stats import norm
from scipy import linalg
import quadprog
from scipy.stats import rankdata
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, euclidean_distances
from sklearn.model_selection import GridSearchCV, cross_val_predict, StratifiedKFold
from sklearn.utils import check_X_y
from quantificationlib.utils import solve_hd, is_pd, nearest_pd
from sklearn.utils.validation import check_array, check_consistent_length


class BaseQuantifier(six.with_metaclass(ABCMeta, BaseEstimator)):
    pass


class UsingClassifiers(BaseQuantifier):
    def __init__(self, estimator_train=None, estimator_test=None, needs_predictions_train=True,
                 probabilistic_predictions=True, verbose=0, **kwargs):
        super(UsingClassifiers, self).__init__(**kwargs)
        # init attributes
        self.estimator_train = estimator_train
        self.estimator_test = estimator_test
        self.needs_predictions_train = needs_predictions_train
        self.probabilistic_predictions = probabilistic_predictions
        self.verbose = verbose
        # computed attributes
        self.predictions_test_ = None
        self.predictions_train_ = None
        self.classes_ = None
        self.y_ext_ = None

    def fit(self, X, y, predictions_train=None):
        self.classes_ = np.unique(y)

        if self.needs_predictions_train and self.estimator_train is None and predictions_train is None:
            raise ValueError("estimator_train or predictions_train must be not None "
                             "with objects of class %s", self.__class__.__name__)

        # Fit estimators if they are not already fitted
        if self.estimator_train is not None:
            if self.verbose > 0:
                print('Class %s: Fitting estimator for training distribution...' % self.__class__.__name__, end='')
            # we need to fit the estimator for the training distribution
            # we check if the estimator is trained or not
            try:
                self.estimator_train.predict(X[0:1, :].reshape(1, -1))
                if self.verbose > 0:
                    print('it was already fitted')

            except NotFittedError:

                X, y = check_X_y(X, y, accept_sparse=True)

                self.estimator_train.fit(X, y)

                if self.verbose > 0:
                    print('fitted')

        if self.estimator_test is not None:
            if self.verbose > 0:
                print('Class %s: Fitting estimator for testing distribution...' % self.__class__.__name__, end='')

            # we need to fit the estimator for the testing distribution
            # we check if the estimator is trained or not
            try:
                self.estimator_test.predict(X[0:1, :].reshape(1, -1))
                if self.verbose > 0:
                    print('it was already fitted')

            except NotFittedError:

                X, y = check_X_y(X, y, accept_sparse=True)

                self.estimator_test.fit(X, y)

                if self.verbose > 0:
                    print('fitted')

        # Compute predictions_train_
        if self.verbose > 0:
            print('Class %s: Computing predictions for training distribution...' % self.__class__.__name__, end='')

        if self.needs_predictions_train:
            if predictions_train is not None:
                if self.probabilistic_predictions:
                    self.predictions_train_ = predictions_train
                else:
                    self.predictions_train_ = UsingClassifiers.__probs2crisps(predictions_train, self.classes_)
            else:
                if self.probabilistic_predictions:
                    self.predictions_train_ = self.estimator_train.predict_proba(X)
                else:
                    self.predictions_train_ = self.estimator_train.predict(X)

            # Compute y_ext_
            if len(y) == len(self.predictions_train_):
                self.y_ext_ = y
            else:
                self.y_ext_ = np.tile(y, len(self.predictions_train_) // len(y))

        if self.verbose > 0:
            print('done')

        return self

    def predict(self, X, predictions_test=None):
        if self.estimator_test is None and predictions_test is None:
            raise ValueError("estimator_test or predictions_test must be not None "
                             "to compute a prediction with objects of class %s", self.__class__.__name__)

        if self.verbose > 0:
            print('Class %s: Computing predictions for testing distribution...' % self.__class__.__name__, end='')

        # At least one between estimator_test and predictions_test is not None
        if predictions_test is not None:
            if self.probabilistic_predictions:
                self.predictions_test_ = predictions_test
            else:
                self.predictions_test_ = UsingClassifiers.__probs2crisps(predictions_test, self.classes_)
        else:
            check_array(X, accept_sparse=True)
            if self.probabilistic_predictions:
                self.predictions_test_ = self.estimator_test.predict_proba(X)
            else:
                self.predictions_test_ = self.estimator_test.predict(X)

        if self.verbose > 0:
            print('done')

        return self

    @staticmethod
    def __probs2crisps(preds, labels):
        if len(preds) == 0:
            return preds
        if preds.ndim == 1 or preds.shape[1] == 1:
            #  binary problem
            if preds.ndim == 1:
                preds_mod = np.copy(preds)
            else:
                preds_mod = np.copy(preds.squeeze())
            if isinstance(preds_mod[0], np.float):
                # it contains probs
                preds_mod[preds_mod >= 0.5] = 1
                preds_mod[preds_mod < 0.5] = 0
                return preds_mod.astype(int)
            else:
                return preds_mod
        else:
            # multiclass problem
            return labels.take(preds.argmax(axis=1), axis=0)

class CC(UsingClassifiers):
    def __init__(self, estimator_test=None, verbose=0):
        super(CC, self).__init__(estimator_test=estimator_test,
                                 needs_predictions_train=False, probabilistic_predictions=False, verbose=verbose)

    def fit(self, X, y, predictions_train=None):
        super().fit(X, y, predictions_train=[])

        return self

    def predict(self, X, predictions_test=None):
        super().predict(X, predictions_test=predictions_test)

        if self.verbose > 0:
            print('Class %s: Computing prevalences for testing distribution...' % self.__class__.__name__, end='')

        n_classes = len(self.classes_)

        freq = np.zeros((n_classes, 1))
        for n_cls, cls in enumerate(self.classes_):
            freq[n_cls, 0] = np.equal(self.predictions_test_, cls).sum()

        prevalences = freq / float(len(self.predictions_test_))

        if self.verbose > 0:
            print('done')

        return np.squeeze(prevalences)

class AC(UsingClassifiers):
    def __init__(self, estimator_train=None, estimator_test=None, distance='HD', verbose=0):
        super(AC, self).__init__(estimator_train=estimator_train, estimator_test=estimator_test,
                                 needs_predictions_train=True, probabilistic_predictions=False, verbose=verbose)
        self.distance = distance
        # confusion matrix
        self.cm_ = None
        # variables for solving the optimization problem when n_classes > 2 and distance = 'L2'
        self.G_ = None
        self.C_ = None
        self.b_ = None
        self.problem_ = None

    def fit(self, X, y, predictions_train=None):
        super().fit(X, y, predictions_train=predictions_train)

        if self.verbose > 0:
            print('Class %s: Estimating confusion matrix for training distribution...' % self.__class__.__name__,
                  end='')

        #  estimating the confusion matrix
        cm = confusion_matrix(self.y_ext_, self.predictions_train_, labels=self.classes_)
        #  normalizing cm by row
        self.cm_ = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # binary:  [[1-fpr  fpr]
        #                                                                          [1-tpr  tpr]]

        if len(self.classes_) > 2 and self.distance == 'L2':
            self.G_, self.C_, self.b_ = compute_l2_param_train(self.cm_.T, self.classes_)

        if self.verbose > 0:
            print('done')

        self.problem_ = None

        return self

    def predict(self, X, predictions_test=None):
        super().predict(X, predictions_test=predictions_test)

        if self.verbose > 0:
            print('Class %s: Computing prevalences for testing distribution...' % self.__class__.__name__, end='')

        n_classes = len(self.classes_)

        freq = np.zeros((n_classes, 1))
        for n_cls, cls in enumerate(self.classes_):
            freq[n_cls, 0] = np.equal(self.predictions_test_, cls).sum()

        prevalences_0 = freq / float(len(self.predictions_test_))

        if n_classes == 2:
            if np.abs((self.cm_[1, 1] - self.cm_[0, 1])) > 0.001:
                p = (prevalences_0[1] - self.cm_[0, 1]) / (self.cm_[1, 1] - self.cm_[0, 1])
                prevalences = [1-p, p]
            else:
                prevalences = prevalences_0

            # clipping the prevalences according to (Forman 2008)
            prevalences = np.clip(prevalences, 0, 1)

            if np.sum(prevalences) > 0:
                prevalences = prevalences / float(np.sum(prevalences))

            prevalences = prevalences.squeeze()
        else:
            try:
                inv_cm = linalg.inv(self.cm_.T)
                prevalences = inv_cm.dot(prevalences_0)

                prevalences[prevalences < 0] = 0
                prevalences = prevalences / sum(prevalences)

            except np.linalg.LinAlgError:
                #  inversion fails, looking for a solution that optimizes the selected distance
                if self.distance == 'HD':
                    self.problem_, prevalences = solve_hd(train_distrib=self.cm_.T, test_distrib=prevalences_0,
                                                          n_classes=n_classes, problem=self.problem_)
                elif self.distance == 'L2':
                    prevalences = solve_l2(train_distrib=self.cm_.T, test_distrib=prevalences_0,
                                           G=self.G_, C=self.C_, b=self.b_)
                elif self.distance == 'L1':
                    self.problem_, prevalences = solve_l1(train_distrib=self.cm_.T, test_distrib=prevalences_0,
                                                          n_classes=n_classes, problem=self.problem_)
                else:
                    raise ValueError('Class %s": distance function not supported', self.__class__.__name__)

        if self.verbose > 0:
            print('done')

        return prevalences.squeeze()

class PAC(UsingClassifiers):
    def __init__(self, estimator_test=None, estimator_train=None, distance='L2', verbose=0):
        super(PAC, self).__init__(estimator_train=estimator_train, estimator_test=estimator_test,
                                  needs_predictions_train=True, probabilistic_predictions=True, verbose=verbose)
        self.distance = distance
        # confusion matrix with average probabilities
        self.cm_ = None
        # variables for solving the optimization problem when n_classes > 2 and distance = 'L2'
        self.G_ = None
        self.C_ = None
        self.b_ = None
        self.problem_ = None

    def fit(self, X, y, predictions_train=None):
        super().fit(X, y, predictions_train=predictions_train)

        if self.verbose > 0:
            print('Class %s: Estimating average probabilities for training distribution...'
                  % self.__class__.__name__, end='')

        n_classes = len(self.classes_)
        # estimating the confusion matrix
        # average probabilty distribution for each class
        self.cm_ = np.zeros((n_classes, n_classes))
        for n_cls, cls in enumerate(self.classes_):
            self.cm_[n_cls] = np.mean(self.predictions_train_[self.y_ext_ == cls], axis=0)

        if self.verbose > 0:
            print('done')

        self.problem_ = None

        return self

    def predict(self, X, predictions_test=None):
        super().predict(X, predictions_test=predictions_test)

        if self.verbose > 0:
            print('Class %s: Computing prevalences for testing distribution...' % self.__class__.__name__, end='')

        n_classes = len(self.classes_)
        prevalences_0 = np.mean(self.predictions_test_, axis=0)

        if n_classes == 2:
            if np.abs(self.cm_[1, 1] - self.cm_[0, 1]) > 0.001:
                p = (prevalences_0[1] - self.cm_[0, 1]) / (self.cm_[1, 1] - self.cm_[0, 1])
                prevalences = [1 - p, p]
            else:
                prevalences = prevalences_0
            # prevalences = np.linalg.solve(self.cm_.T, prevalences_0)

            # clipping the prevalences according to (Forman 2008)
            prevalences = np.clip(prevalences, 0, 1)

            if np.sum(prevalences) > 0:
                prevalences = prevalences / float(np.sum(prevalences))

            prevalences = prevalences.squeeze()
        else:
            try:
                inv_cm = linalg.inv(self.cm_.T)
                prevalences = inv_cm.dot(prevalences_0)

                prevalences[prevalences < 0] = 0
                prevalences = prevalences / sum(prevalences)

            except np.linalg.LinAlgError:
                # inversion fails, looking for a solution that optimizes the selected distance
                if self.distance == 'HD':
                    self.problem_, prevalences = solve_hd(train_distrib=self.cm_.T, test_distrib=prevalences_0,
                                                          n_classes=n_classes, problem=self.problem_)
                elif self.distance == 'L2':
                    prevalences = solve_l2(train_distrib=self.cm_.T, test_distrib=prevalences_0,
                                           G=self.G_, C=self.C_, b=self.b_)
                elif self.distance == 'L1':
                    self.problem_, prevalences = solve_l1(train_distrib=self.cm_.T, test_distrib=prevalences_0,
                                                          n_classes=n_classes, problem=self.problem_)
                else:
                    raise ValueError('Class %s": distance function not supported', self.__class__.__name__)

        if self.verbose > 0:
            print('done')

        return prevalences

class PCC(UsingClassifiers):
    """ Multiclass Probabilistic Classify And Count method.
        prevalence (class_i) = sum_{x in T} P( h(x) == class_i | x )
        This class works in two different ways:
        1) An estimator is used to classify the examples of the testing bag (the estimator can be already trained)
        2) You can directly provide the predictions for the examples in the predict method. This is useful
           for synthetic/artificial experiments
        Parameters
        ----------
        estimator_test : estimator object (default=None)
            An estimator object implementing `fit` and `predict` methods. It is used to classify the testing examples
        verbose : int, optional, (default=0)
            The verbosity level. The default value, zero, means silent mode
        Attributes
        ----------
        estimator_test : estimator
            Estimator used to classify the examples of the testing bag
        predictions_test_ : ndarray, shape (n_examples, n_classes)
            Probabilistic predictions of the examples in the testing bag
        estimator_train : None. (Not used)
        predictions_train_ : None. (Not used)
        needs_predictions_train : bool, False
            It is False because PCC quantifiers do not need to estimate the training distribution
        probabilistic_predictions : bool, True
             This means that predictions_test_ contains probabilistic predictions
        classes_ : ndarray, shape (n_classes, )
            Class labels
        y_ext_ : ndarray, shape(n_examples, )
            True labels of the training set
        verbose : int
            The verbosity level
        Notes
        -----
        Notice that at least one between estimator_test and predictions_test must be not None. If both are None a
        ValueError exception will be raised. If both are not None, predictions_test is used.
        References
        ----------
        Antonio Bella, Cèsar Ferri, José Hernández-Orallo, and María José Ramírez-Quintana. 2010. Quantification
        via probability estimators. In Proceedings of the IEEE International Conference on Data Mining (ICDM’10).
        IEEE, 737–742.
    """
    def __init__(self, estimator_test=None, verbose=0):
        super(PCC, self).__init__(estimator_test=estimator_test,
                                  needs_predictions_train=False, probabilistic_predictions=True, verbose=verbose)

    def fit(self, X, y, predictions_train=None):
        """ Fit the estimator for the testing bags when needed. The method checks whether the estimator is trained or
            not calling the predict method
            Parameters
            ----------
            X : (sparse) array-like, shape (n_examples, n_features)
                Data
            y : (sparse) array-like, shape (n_examples, )
                True classes
            predictions_train : Not used
        """
        super().fit(X, y, predictions_train=[])

        return self

    def predict(self, X, predictions_test=None):
        """ Predict the class distribution of a testing bag
            The prevalence for each class is the average probability for such class
            prevalence (class_i) = sum_{x in T} P( h(x) == class_i | x )
            Parameters
            ----------
            X : (sparse) array-like, shape (n_examples, n_features)
                Data
            predictions_test : ndarray, shape (n_examples, n_classes) (default=None)
                They must be probabilities (the estimator used must have a predict_proba method)
                If predictions_test is not None they are copied on predictions_test_ and used.
                If predictions_test is None, predictions for the testing examples are computed using the `predict`
                method of estimator_test (it must be an actual estimator)
            Raises
            ------
            ValueError
                When estimator_test and predictions_test are both None
            Returns
            -------
            prevalences : An ndarray, shape(n_classes, ) with the prevalence for each class
        """
        super().predict(X, predictions_test=predictions_test)

        if self.verbose > 0:
            print('Class %s: Computing prevalences for testing distribution...' % self.__class__.__name__, end='')

        prevalences = np.mean(self.predictions_test_, axis=0)

        if self.verbose > 0:
            print('done')

        return prevalences


def check_prevalences(p_true, p_pred):
    check_consistent_length(p_true, p_pred)
    p_true = check_array(p_true, ensure_2d=False)
    p_pred = check_array(p_pred, ensure_2d=False)

    if p_true.ndim == 1:
        p_true = p_true.reshape((-1, 1))

    if p_pred.ndim == 1:
        p_pred = p_pred.reshape((-1, 1))

    if p_true.shape[1] != p_pred.shape[1]:
        raise ValueError("p_true and p_pred have different length")

    return p_true, p_pred

def mean_absolute_error(p_true, p_pred):
    p_true, p_pred = check_prevalences(p_true, p_pred)
    return np.mean(np.abs(p_pred - p_true))

class EM(UsingClassifiers):
 
    def __init__(self, estimator_train=None, estimator_test=None, verbose=0, epsilon=1e-4, max_iter=1000):
        super(EM, self).__init__(estimator_train=estimator_train, estimator_test=estimator_test,
                                 needs_predictions_train=True, probabilistic_predictions=True, verbose=verbose)
        self.epsilon_ = epsilon
        self.max_iter_ = max_iter
        self.prevalences_train_ = None

    def fit(self, X, y, predictions_train=None):

        super().fit(X, y, predictions_train=predictions_train)

        n_classes = len(self.classes_)

        freq = np.zeros(n_classes)
        for n_cls, cls in enumerate(self.classes_):
            freq[n_cls] = np.equal(y, cls).sum()

        self.prevalences_train_ = freq / float(len(y))

        return self

    def predict(self, X, predictions_test=None):

        super().predict(X, predictions_test=predictions_test)

        if self.verbose > 0:
            print('Class %s: Estimating prevalences for testing distribution...' % self.__class__.__name__, end='')

        n_classes = len(self.classes_)
        iterations = 0
        prevalences = np.copy(self.prevalences_train_)
        prevalences_prev = np.ones(n_classes)

        while iterations < self.max_iter_ and (mean_absolute_error(prevalences, prevalences_prev) > self.epsilon_
                                               or iterations < 10):

            nonorm_posteriors = np.multiply(self.predictions_test_, np.divide(prevalences, self.prevalences_train_))

            posteriors = np.divide(nonorm_posteriors, nonorm_posteriors.sum(axis=1, keepdims=True))

            prevalences_prev = prevalences
            prevalences = posteriors.mean(0)

            iterations = iterations + 1

        if self.verbose > 0:
            if iterations < self.max_iter_:
                print('done')
            else:
                print('done but it might have not converged, max_iter reached')

        return prevalences