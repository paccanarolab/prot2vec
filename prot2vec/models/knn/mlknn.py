from sklearn.base import ClassifierMixin, MultiOutputMixin
from sklearn.neighbors import NearestNeighbors
import numpy as np
import scipy.sparse as sparse

"""
reimplementation of MLkNN, taking heavily from skmultilearn:
http://scikit.ml/api/skmultilearn.adapt.mlknn.html#multilabel-k-nearest-neighbours

most of the code is taken verbatim from that project, that as of December 2021 seems
to be abandoned.
"""

SPARSE_FORMAT_TO_CONSTRUCTOR = {
    "bsr": sparse.bsr_matrix,
    "coo": sparse.coo_matrix,
    "csc": sparse.csc_matrix,
    "csr": sparse.csr_matrix,
    "dia": sparse.dia_matrix,
    "dok": sparse.dok_matrix,
    "lil": sparse.lil_matrix
}

def get_matrix_in_format(original_matrix, matrix_format):
    """Converts matrix to format

    Parameters
    ----------

    original_matrix : np.matrix or scipy matrix or np.array of np. arrays
        matrix to convert

    matrix_format : string
        format

    Returns
    -------

    matrix : scipy matrix
        matrix in given format
    """
    if isinstance(original_matrix, np.ndarray):
        return SPARSE_FORMAT_TO_CONSTRUCTOR[matrix_format](original_matrix)

    if original_matrix.getformat() == matrix_format:
        return original_matrix

    return original_matrix.asformat(matrix_format)


class MultiLabelKNeighborsClassifier(ClassifierMixin, MultiOutputMixin):

    """reimplementation of MLkNN, taking heavily from skmultilearn:
    http://scikit.ml/api/skmultilearn.adapt.mlknn.html#multilabel-k-nearest-neighbours

    The main difference is that the interface to the underlying
    sklearn.neighbors.NearestNeighbors object is exposed, so that it can be used with
    precomputed distances.


    References
    ----------

    If you use this classifier please cite the original paper introducing the method:

    .. code :: bibtex

        @article{zhang2007ml,
          title={ML-KNN: A lazy learning approach to multi-label learning},
          author={Zhang, Min-Ling and Zhou, Zhi-Hua},
          journal={Pattern recognition},
          volume={40},
          number={7},
          pages={2038--2048},
          year={2007},
          publisher={Elsevier}
        }
    """

    def __init__(
            self,
            s = 1.0,
            *,
            n_neighbors=5,
            radius=1.0,
            algorithm="auto",
            leaf_size=30,
            metric="minkowski",
            p=2,
            metric_params=None,
            n_jobs=None,
            ignore_first_neighbours=True
    ):
        self.s = s
        self.nn_ = NearestNeighbors(
            n_neighbors=n_neighbors,
            radius=radius,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs
        )
        self.n_neighbors = n_neighbors
        self.ignore_first_neighbours = ignore_first_neighbours

    def _compute_prior(self, y):
        """Helper function to compute for the prior probabilities

        Parameters
        ----------
        y : numpy.ndarray or scipy.sparse
            the training labels

        Returns
        -------
        numpy.ndarray
            the prior probability given true
        numpy.ndarray
            the prior probability given false
        """
        prior_prob_true = np.array((self.s + y.sum(axis=0)) / (self.s * 2 + self._num_instances))[0]
        prior_prob_false = 1 - prior_prob_true

        return (prior_prob_true, prior_prob_false)

    def _compute_cond(self, X, y):
        """Helper function to compute for the posterior probabilities

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse
            input features, can be a dense or sparse matrix of size
            :code:`(n_samples, n_features)`
        y : numpy.ndaarray or scipy.sparse {0,1}
            binary indicator matrix with label assignments.

        Returns
        -------
        numpy.ndarray
            the posterior probability given true
        numpy.ndarray
            the posterior probability given false
        """

        self.nn_.fit(X)
        c = sparse.lil_matrix((self._num_labels, self.n_neighbors + 1), dtype='i8')
        cn = sparse.lil_matrix((self._num_labels, self.n_neighbors + 1), dtype='i8')

        label_info = get_matrix_in_format(y, 'dok')

        neighbors = [a[self.ignore_first_neighbours:] for a in
                     self.nn_.kneighbors(X,
                                         n_neighbors=self.n_neighbors + self.ignore_first_neighbours,
                                         return_distance=False)]

        for instance in range(self._num_instances):
            deltas = label_info[neighbors[instance], :].sum(axis=0)
            for label in range(self._num_labels):
                if label_info[instance, label] == 1:
                    c[label, deltas[0, label]] += 1
                else:
                    cn[label, deltas[0, label]] += 1

        c_sum = c.sum(axis=1)
        cn_sum = cn.sum(axis=1)

        cond_prob_true = sparse.lil_matrix((self._num_labels, self.n_neighbors + 1), dtype='float')
        cond_prob_false = sparse.lil_matrix((self._num_labels, self.n_neighbors + 1), dtype='float')
        for label in range(self._num_labels):
            for neighbor in range(self.n_neighbors + 1):
                cond_prob_true[label, neighbor] = (self.s + c[label, neighbor]) / (
                        self.s * (self.n_neighbors + 1) + c_sum[label, 0])
                cond_prob_false[label, neighbor] = (self.s + cn[label, neighbor]) / (
                        self.s * (self.n_neighbors + 1) + cn_sum[label, 0])
        return cond_prob_true, cond_prob_false

    def fit(self, X, y):
        """Fit classifier with training data

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse
            input features, can be a dense or sparse matrix of size
            :code:`(n_samples, n_features)`
        y : numpy.ndaarray or scipy.sparse {0,1}
            binary indicator matrix with label assignments.

        Returns
        -------
        self
            fitted instance of self
        """

        self._label_cache = get_matrix_in_format(y, 'lil')
        self._num_instances = self._label_cache.shape[0]
        self._num_labels = self._label_cache.shape[1]
        # Computing the prior probabilities
        self._prior_prob_true, self._prior_prob_false = self._compute_prior(self._label_cache)
        # Computing the posterior probabilities
        self._cond_prob_true, self._cond_prob_false = self._compute_cond(X, self._label_cache)
        return self

    def predict(self, X):
        """Predict labels for X

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csc_matrix
            input features of shape :code:`(n_samples, n_features)`

        Returns
        -------
        scipy.sparse matrix of int
            binary indicator matrix with label assignments with shape
            :code:`(n_samples, n_labels)`
        """

        result = sparse.lil_matrix((X.shape[0], self._num_labels), dtype='i8')
        neighbors = [a[self.ignore_first_neighbours:] for a in
                     self.nn_.kneighbors(X,
                                         n_neighbors=self.n_neighbors + self.ignore_first_neighbours,
                                         return_distance=False)]
        for instance in range(X.shape[0]):
            deltas = self._label_cache[neighbors[instance],].sum(axis=0)

            for label in range(self._num_labels):
                p_true = self._prior_prob_true[label] * self._cond_prob_true[label, deltas[0, label]]
                p_false = self._prior_prob_false[label] * self._cond_prob_false[label, deltas[0, label]]
                result[instance, label] = int(p_true >= p_false)

        return result

    def predict_proba(self, X):
        """Predict probabilities of label assignments for X

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csc_matrix
            input features of shape :code:`(n_samples, n_features)`

        Returns
        -------
        scipy.sparse matrix of int
            binary indicator matrix with label assignment probabilities
            with shape :code:`(n_samples, n_labels)`
        """
        result = sparse.lil_matrix((X.shape[0], self._num_labels), dtype='float')
        neighbors = [a[self.ignore_first_neighbours:] for a in
                     self.nn_.kneighbors(X,
                                         n_neighbors=self.n_neighbors + self.ignore_first_neighbours,
                                         return_distance=False)]
        for instance in range(X.shape[0]):
            deltas = self._label_cache[neighbors[instance],].sum(axis=0)

            for label in range(self._num_labels):
                p_true = self._prior_prob_true[label] * self._cond_prob_true[label, deltas[0, label]]
                result[instance, label] = p_true

        return result