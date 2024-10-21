import torch
from functools import partial

from typing import Optional, Tuple



class IncrementalPCA:
    """
    An implementation of Incremental Principal Components Analysis (IPCA) that leverages PyTorch for GPU acceleration.
    Adapted from https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/decomposition/_incremental_pca.py

    This class provides methods to fit the model on data incrementally in batches, and to transform new data
    based on the principal components learned during the fitting process.

    Attributes:
        n_components (int, optional): Number of components to keep. If `None`, it's set to the minimum of the
                                number of samples and features. Defaults to None.
        copy (bool): If False, input data will be overwritten. Defaults to True.
        batch_size (int, optional): The number of samples to use for each batch. Only needed if self.fit is called.
                                If `None`, it's inferred from the data and set to `5 * n_features`. Defaults to None.
        svd_driver (str, optional): name of the cuSOLVER method to be used for torch.linalg.svd. This keyword 
                                argument only works on CUDA inputs. Available options are: None, gesvd, gesvdj,
                                and gesvda. Defaults to None.
        lowrank (bool, optional): Whether to use torch.svd_lowrank instead of torch.linalg.svd which can be faster.
                                Defaults to False.
        lowrank_q (int, optional): For an adequate approximation of n_components, this parameter defaults to 
                                n_components * 2.
        lowrank_niter (int, optional): Number of subspace iterations to conduct for torch.svd_lowrank.
                                Defaults to 4. 
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        copy: Optional[bool] = True,
        batch_size: Optional[int] = None,
        svd_driver: Optional[str] = None,
        lowrank: bool = False,
        lowrank_q: Optional[int] = None,
        lowrank_niter: int = 4
    ):
        self.n_components_ = n_components
        self.copy = copy
        self.batch_size = batch_size

        if lowrank:
            if lowrank_q is None:
                assert n_components is not None, "n_components must be specified when using lowrank mode with lowrank_q=None."
                lowrank_q = n_components * 2
            assert lowrank_q >= n_components, "lowrank_q must be greater than or equal to n_components."
            def svd_fn(X):
                U, S, V = torch.svd_lowrank(X, q=lowrank_q, niter=lowrank_niter)
                return U, S, V.mH # V is returned as a conjugate transpose
            self._svd_fn = svd_fn

        else:
            self._svd_fn = partial(torch.linalg.svd, full_matrices=False, driver=svd_driver)
        

    def _validate_data(self, X, dtype=torch.float32) -> torch.Tensor:
        """
        Validates and converts the input data `X` to the appropriate tensor format.

        Args:
            X (torch.Tensor): Input data.
            dtype (torch.dtype, optional): Desired data type for the tensor. Defaults to torch.float32.

        Returns:
            torch.Tensor: Converted to appropriate format.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=dtype)
        elif self.copy:
            X = X.clone()

        if X.dtype != dtype:
            X = X.to(dtype)

        return X

    @staticmethod
    def _incremental_mean_and_var(X, last_mean, last_variance, last_sample_count) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the incremental mean and variance for the data `X`.

        Args:
            X (torch.Tensor): The batch input data tensor with shape (n_samples, n_features).
            last_mean (torch.Tensor): The previous mean tensor with shape (n_features,).
            last_variance (torch.Tensor): The previous variance tensor with shape (n_features,).
            last_sample_count (torch.Tensor): The count tensor of samples processed before the current batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Updated mean, variance tensors, and total sample count.
        """
        if X.shape[0] == 0:
            return last_mean, last_variance, last_sample_count

        if last_sample_count > 0:
            assert (
                last_mean is not None
            ), "last_mean should not be None if last_sample_count > 0."
            assert (
                last_variance is not None
            ), "last_variance should not be None if last_sample_count > 0."

        new_sample_count = torch.tensor([X.shape[0]], device=X.device)
        updated_sample_count = last_sample_count + new_sample_count

        if last_mean is None:
            last_sum = torch.zeros(X.shape[1], dtype=torch.float64, device=X.device)
        else:
            last_sum = last_mean * last_sample_count

        new_sum = X.sum(dim=0, dtype=torch.float64)

        updated_mean = (last_sum + new_sum) / updated_sample_count

        T = new_sum / new_sample_count
        temp = X - T
        correction = temp.sum(dim=0, dtype=torch.float64).square()
        temp.square_()
        new_unnormalized_variance = temp.sum(dim=0, dtype=torch.float64)
        new_unnormalized_variance -= correction / new_sample_count
        if last_variance is None:
            updated_variance = new_unnormalized_variance / updated_sample_count
        else:
            last_unnormalized_variance = last_variance * last_sample_count
            last_over_new_count = last_sample_count.double() / new_sample_count
            updated_unnormalized_variance = (
                last_unnormalized_variance
                + new_unnormalized_variance
                + last_over_new_count
                / updated_sample_count
                * (last_sum / last_over_new_count - new_sum).square()
            )
            updated_variance = updated_unnormalized_variance / updated_sample_count

        return updated_mean, updated_variance, updated_sample_count

    @staticmethod
    def _svd_flip(u, v, u_based_decision=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adjusts the signs of the singular vectors from the SVD decomposition for deterministic output.

        This method ensures that the output remains consistent across different runs.

        Args:
            u (torch.Tensor): Left singular vectors tensor.
            v (torch.Tensor): Right singular vectors tensor.
            u_based_decision (bool, optional): If True, uses the left singular vectors to determine the sign flipping. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Adjusted left and right singular vectors tensors.
        """
        if u_based_decision:
            max_abs_cols = torch.argmax(torch.abs(u), dim=0)
            signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
        else:
            max_abs_rows = torch.argmax(torch.abs(v), dim=1)
            signs = torch.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs[:u.shape[1]].view(1, -1)
        v *= signs.view(-1, 1)
        return u, v

    def fit(self, X, check_input=True):
        """
        Fits the model with data `X` using minibatches of size `batch_size`.

        Args:
            X (torch.Tensor): The input data tensor with shape (n_samples, n_features).
            check_input (bool, optional): If True, validates the input. Defaults to True.

        Returns:
            IncrementalPCA: The fitted IPCA model.
        """
        if check_input:
            X = self._validate_data(X)
        n_samples, n_features = X.shape
        if self.batch_size is None:
            self.batch_size_ = 5 * n_features
        else:
            self.batch_size_ = self.batch_size

        for start in range(0, n_samples, self.batch_size_):
            end = min(start + self.batch_size_, n_samples)
            X_batch = X[start:end]
            self.partial_fit(X_batch, check_input=False)

        return self

    def partial_fit(self, X, check_input=True):
        """
        Incrementally fits the model with batch data `X`.

        Args:
            X (torch.Tensor): The batch input data tensor with shape (n_samples, n_features).
            check_input (bool, optional): If True, validates the input. Defaults to True.

        Returns:
            IncrementalPCA: The updated IPCA model after processing the batch.
        """
        first_pass = not hasattr(self, "components_")

        if check_input:
            X = self._validate_data(X)
        n_samples, n_features = X.shape

        # Initialize attributes to avoid errors during the first call to partial_fit
        if first_pass:
            self.mean_ = None  # Will be initialized properly in _incremental_mean_and_var based on data dimensions
            self.var_ = None  # Will be initialized properly in _incremental_mean_and_var based on data dimensions
            self.n_samples_seen_ = torch.tensor([0], device=X.device)
            if not self.n_components_:
                self.n_components_ = min(n_samples, n_features)

        col_mean, col_var, n_total_samples = self._incremental_mean_and_var(
            X, self.mean_, self.var_, self.n_samples_seen_
        )

        if first_pass:
            X -= col_mean
        else:
            col_batch_mean = torch.mean(X, dim=0)
            X -= col_batch_mean
            mean_correction_factor = torch.sqrt(
                (self.n_samples_seen_.double() / n_total_samples) * n_samples
            )
            mean_correction = mean_correction_factor * (self.mean_ - col_batch_mean)
            X = torch.vstack(
                (
                    self.singular_values_.view((-1, 1)) * self.components_,
                    X,
                    mean_correction,
                )
            )

        U, S, Vt = self._svd_fn(X)
        U, Vt = self._svd_flip(U, Vt, u_based_decision=False)
        explained_variance = S**2 / (n_total_samples - 1)
        explained_variance_ratio = S**2 / torch.sum(col_var * n_total_samples)

        self.n_samples_seen_ = n_total_samples
        self.components_ = Vt[:self.n_components_]
        self.singular_values_ = S[:self.n_components_]
        self.mean_ = col_mean
        self.var_ = col_var
        self.explained_variance_ = explained_variance[:self.n_components_]
        self.explained_variance_ratio_ = explained_variance_ratio[:self.n_components_]
        if self.n_components_ not in (n_samples, n_features):
            self.noise_variance_ = explained_variance[self.n_components_:].mean()
        else:
            self.noise_variance_ = torch.tensor(0.0, device=X.device)
        return self

    def transform(self, X) -> torch.Tensor:
        """
        Applies dimensionality reduction to `X`.

        The input data `X` is projected on the first principal components previously extracted from a training set.

        Args:
            X (torch.Tensor): New data tensor with shape (n_samples, n_features) to be transformed.

        Returns:
            torch.Tensor: Transformed data tensor with shape (n_samples, n_components).
        """
        X -= self.mean_
        return torch.mm(X, self.components_.T)