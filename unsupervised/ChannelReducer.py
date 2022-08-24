""" Taken from original repo (https://github.com/zhangrh93/InvertibleCE) """

"""Helper for using sklearn.decomposition on high-dimensional tensors.
Provides ChannelReducer, a wrapper around sklearn.decomposition to help them
apply to arbitrary rank tensors. It saves lots of annoying reshaping.
"""

import numpy as np
import sklearn.decomposition
import sklearn.cluster
import tensorly as tl
from tensorly.decomposition import non_negative_tucker_hals
import tensorly

from sklearn.base import BaseEstimator

ALGORITHM_NAMES = {}
for name in dir(sklearn.decomposition):
    obj = sklearn.decomposition.__getattribute__(name)
    if isinstance(obj, type) and issubclass(obj, BaseEstimator):
        ALGORITHM_NAMES[name] = "decomposition"
for name in dir(sklearn.cluster):
    obj = sklearn.cluster.__getattribute__(name)
    if isinstance(obj, type) and issubclass(obj, BaseEstimator):
        ALGORITHM_NAMES[name] = "cluster"
# add tucker decomposition
ALGORITHM_NAMES["NTD"] = "3d_decomposition"


class ChannelDecompositionReducer(object):
    def __init__(self, n_components=3, reduction_alg="NMF", **kwargs):

        if not isinstance(n_components, int):
            raise ValueError("n_components must be an int, not '%s'." % n_components)

        # Defensively look up reduction_alg if it is a string and give useful errors.
        algorithm_map = {}
        for name in dir(sklearn.decomposition):
            obj = sklearn.decomposition.__getattribute__(name)
            if isinstance(obj, type) and issubclass(obj, BaseEstimator):
                algorithm_map[name] = obj
        if isinstance(reduction_alg, str):
            if reduction_alg in algorithm_map:
                reduction_alg = algorithm_map[reduction_alg]
            else:
                raise ValueError(
                    "Unknown dimensionality reduction method '%s'." % reduction_alg
                )

        self.n_components = n_components
        self._reducer = reduction_alg(n_components=n_components, **kwargs)
        self._is_fit = False
        self.reducer_conv = None

    def _apply_flat(cls, f, acts):
        """ flat the input matrix A to (something x c), 
        run the NMF on that to produce (W,H) of shapes (something x c') and (c' x c) ,
        reshape W according to the initial shape of A, except the last dimension that stays c' as it was in W.
         """
        orig_shape = acts.shape
        acts_flat = acts.reshape([-1, acts.shape[-1]])
        new_flat = f(acts_flat)
        if not isinstance(new_flat, np.ndarray):
            return new_flat
        shape = list(orig_shape[:-1]) + [-1]
        return new_flat.reshape(shape)

    def fit(self, acts):
        if hasattr(self._reducer, "partial_fit"):
            res = self._apply_flat(self._reducer.partial_fit, acts)
        else:
            res = self._apply_flat(self._reducer.fit, acts)
        self._is_fit = True
        return res

    def fit_transform(self, acts):
        """Fit a NMF model and return the data tranformed by it.
        This is a wrapper for the sklearn function that flatten and then recostruct the input matrix."""
        # acts is the input matrix to factorize.
        res = self._apply_flat(self._reducer.fit_transform, acts)
        self._is_fit = True
        self.reducer_conv = self._reducer.reconstruction_err_
        return res

    def transform(self, acts):
        """Return the data tranformed by and already fitted NMF model.
        This is a wrapper for the sklearn function that flatten and then recostruct the input matrix."""
        res = self._apply_flat(self._reducer.transform, acts)
        return res

    def inverse_transform(self, acts):
        if hasattr(self._reducer, "inverse_transform"):
            res = self._apply_flat(self._reducer.inverse_transform, acts)
        else:
            res = np.dot(acts, self._reducer.components_)
        return res


class ChannelClusterReducer(object):
    def __init__(self, n_components=3, reduction_alg="KMeans", **kwargs):

        if not isinstance(n_components, int):
            raise ValueError("n_components must be an int, not '%s'." % n_components)

        # Defensively look up reduction_alg if it is a string and give useful errors.
        algorithm_map = {}
        for name in dir(sklearn.cluster):
            obj = sklearn.cluster.__getattribute__(name)
            if isinstance(obj, type) and issubclass(obj, BaseEstimator):
                algorithm_map[name] = obj
        if isinstance(reduction_alg, str):
            if reduction_alg in algorithm_map:
                reduction_alg = algorithm_map[reduction_alg]
            else:
                raise ValueError(
                    "Unknown dimensionality reduction method '%s'." % reduction_alg
                )

        self.n_components = n_components
        self._reducer = reduction_alg(n_clusters=n_components, **kwargs)
        self._is_fit = False

    def _apply_flat(self, f, acts):
        """Utility for applying f to inner dimension of acts.
        Flattens acts into a 2D tensor, applies f, then unflattens so that all
        dimesnions except innermost are unchanged.
        """
        orig_shape = acts.shape
        acts_flat = acts.reshape([-1, acts.shape[-1]])
        new_flat = f(acts_flat)
        if not isinstance(new_flat, np.ndarray):
            return new_flat
        shape = list(orig_shape[:-1]) + [-1]
        new_flat = new_flat.reshape(shape)

        if new_flat.shape[-1] == 1:
            new_flat = new_flat.reshape(-1)
            t_flat = np.zeros([new_flat.shape[0], self.n_components])
            t_flat[np.arange(new_flat.shape[0]), new_flat] = 1
            new_flat = t_flat.reshape(shape)

        return new_flat

    def fit(self, acts):
        if hasattr(self._reducer, "partial_fit"):
            res = self._apply_flat(self._reducer.partial_fit, acts)
        else:
            res = self._apply_flat(self._reducer.fit, acts)
        self._reducer.components_ = self._reducer.cluster_centers_
        self._is_fit = True
        return res

    def fit_predict(self, acts):
        res = self._apply_flat(self._reducer.fit_predict, acts)
        self._reducer.components_ = self._reducer.cluster_centers_
        self._is_fit = True
        return res

    def transform(self, acts):
        res = self._apply_flat(self._reducer.predict, acts)
        return res

    def inverse_transform(self, acts):
        res = np.dot(acts, self._reducer.components_)
        return res


class ChannelTensorDecompositionReducer(object):
    def __init__(self, dimension=3, rank=None, iter_max=1000):
        """Initialize the ChannelDecomposition for tensors.
        Parameters
        ----------
        dimension : int
            the number of dimensions (e.g., 3 for a 3d matrix). Only 3 and 4 are supported.
        rank : list[int]
            list of ranks. Its lenght must match `dimension`.
        iter_max : int
            number of maximum iteration for the tucker decomposition.
        Returns
        """
        if dimension not in [3, 4]:
            raise Exception("Only dimension 3 and 4 are supported.")
        self.dimension = dimension
        if len(rank) != dimension:
            raise Exception("The number of ranks must be the same as dimension.")
        self.rank = rank

        self.precomputed_tensors = None
        self._is_fit = False
        self.trained_shape = None
        self.orig_shape = None
        self._iter_max = iter_max
        self.orig_dataset = None
        self.reducer_conv = None

    def _flat_transpose_pad(self, acts, pad=False):
        """ flat the input matrix A to (c x (h x w) x n),
         """
        # acts is the input matrix to factorize.
        # first transpose it to c x h x w x n (this is necessary because we can't fix last mode in tensorly)
        acts = np.swapaxes(acts, 0, -1)
        if self.dimension == 3:
            # flat the h and w dimension to a single one
            acts = acts.reshape(
                [acts.shape[0], acts.shape[1] * acts.shape[2], acts.shape[3]]
            )
        # zero padd it to have size of the original tucker matrix if not already
        if not pad:
            return acts
        else:
            pad_value = self.trained_shape[-1] - acts.shape[-1]
            if self.dimension == 3:
                pad_width = ((0, 0), (0, 0), (0, pad_value))
            else:  # dimension == 4
                pad_width = ((0, 0), (0, 0), (0, 0), (0, pad_value))
            padded_acts = np.pad(acts, pad_width, mode="constant", constant_values=0)
            return padded_acts

    def _inverse_flat_transpose(self, acts, reduced_channel=True):
        """reshape W according to the initial shape of A."""
        # transpose to put channel at the end and pieces at first
        acts = np.swapaxes(acts, 0, -1)
        # reshape to reconstruct h x w
        if reduced_channel:
            acts = acts.reshape(list(self.orig_shape[:-1]) + [-1])
        else:
            acts = acts.reshape(self.orig_shape)
        return acts

    def fit_transform(self, acts):
        """Fit a tucker model and return the error."""
        # matrix now has the shape n x h x w x c
        self.orig_shape = acts.shape
        self.orig_dataset = acts
        # transpose and flat
        acts_flat = self._flat_transpose_pad(acts)
        # now run tucker decomposition
        print("Running tucker on the matrix of shape", acts_flat.shape)
        print(f"Tucker ranks: {self.rank}")
        tensors, error = non_negative_tucker_hals(
            acts_flat, rank=self.rank, n_iter_max=self._iter_max, return_errors=True
        )
        print("Minimum Tucker error", error[-1])
        normalized_tensors = normalize_tensors(tensors)
        self.precomputed_tensors = normalized_tensors
        self._is_fit = True
        self.trained_shape = acts_flat.shape
        self.reducer_conv = error
        # this is not returning a transformed version of the data. Behaviour is different than NMF
        

    def transform(self, acts):
        # indices = []
        equal_axis = (-1, -2, -3) if self.dimension == 4 else (-1, -2)
        indices = [
            np.all(self.orig_dataset == piece, axis=equal_axis).nonzero()[0][0]
            for piece in acts
        ]
        # for piece in acts:
        #     indices.append(np.all(self.orig_dataset==piece, axis = equal_axis).nonzero()[0][0])
        #     # indices.append((self.orig_dataset == piece).all(axis=-1).nonzero()[0][0])
        output = tensorly.tenalg.multi_mode_dot(
            self.precomputed_tensors.core, self.precomputed_tensors.factors, skip=0
        )
        # reshape and translate the output so we return n x h x w x c'
        output = self._inverse_flat_transpose(output)
        # delete zero-padded pieces
        output = output[indices, :, :, :]
        return output, indices

        # """Return the data tranformed by and already fitted Tucker model."""
        # # acts is not the pianoroll for only some pieces
        # # first flat transpose and pad
        # padded_acts = self._flat_transpose_pad(acts, pad=True)
        # # now tucker decompose with fixed modes (except piece mode)
        # fixed_modes = [0, 1] if self.dimension == 3 else [0, 1, 2]
        # (core, factors), errors = non_negative_tucker_hals(
        #     padded_acts,
        #     rank=self.rank,
        #     return_errors=True,
        #     n_iter_max=self._iter_max,
        #     fixed_modes=fixed_modes,
        #     init=self.precomputed_tensors.tucker_copy(),
        # )
        # print("Minimum fixed Mode Tucker error", errors[-1])
        # # 3 and 2 mode multiplication, skip channel-mode mult
        # output = tensorly.tenalg.multi_mode_dot(core, factors, skip=0)
        # # reshape and translate the output so we return n x h x w x c'
        # output = self._inverse_flat_transpose(output)
        # # delete zero-padded pieces
        # output = output[: acts.shape[0], :, :, :]
        # return output

    def inverse_transform(self, acts, indices):
        if self._is_fit:
            reconstructed = tensorly.tucker_to_tensor(self.precomputed_tensors)
            # transpose and reshape it in original shape n x h x w x c
            reconstructed = self._inverse_flat_transpose(
                reconstructed, reduced_channel=False
            )
            return reconstructed[indices, :, :, :]
        else:
            raise Exception("The Reducer must be fit first")

    # def inverse_transform(self, acts):
    #     # flat transpose
    #     padded_acts = self._flat_transpose_pad(acts, pad=True)
    #     if self._is_fit:
    #         # only step missing to the reconstructed matrix is the 1-mode multiplication
    #         # 3 and 2 mode mult has been performed in transform()
    #         mode1_matrix = self.precomputed_tensors.factors[0]
    #         reconstructed = tensorly.tenalg.mode_dot(padded_acts, mode1_matrix, 0)
    #         # transpose and reshape it in original shape n x h x w x c
    #         reconstructed = self._inverse_flat_transpose(
    #             reconstructed, reduced_channel=False
    #         )
    #         return reconstructed[: acts.shape[0], :, :, :]
    #     else:
    #         raise Exception("The Reducer must be fit first")


def normalize_tensors(tucker_tensor):
    """Returns tucker_tensor with factors normalised to unit length with the normalizing constants absorbed into
    `core`.
    Parameters
    ----------
    tucker_tensor : tl.TuckerTensor or (core, factors)
        core tensor and list of factor matrices
    Returns
    -------
    TuckerTensor((core, factors))
    """
    core, factors = tucker_tensor
    normalized_factors = []
    for i, factor in enumerate(factors):
        scales = tl.norm(factor, axis=0)
        scales_non_zero = tl.where(
            scales == 0, tl.ones(tl.shape(scales), **tl.context(factor)), scales
        )
        core = core * tl.reshape(
            scales, (1,) * i + (-1,) + (1,) * (tl.ndim(core) - i - 1)
        )
        normalized_factors.append(factor / tl.reshape(scales_non_zero, (1, -1)))
    return tl.tucker_tensor.TuckerTensor((core, normalized_factors))
