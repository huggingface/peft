"""Experimental code for adaptive wavelet learning.

See https://arxiv.org/pdf/2004.09569.pdf for more information.
"""

# Inspired by Ripples in Mathematics, Jensen and La Cour-Harbo, Chapter 7.7
from abc import ABC, abstractmethod

import torch


class WaveletFilter(ABC):
    """Interface for learnable wavelets.

    Each wavelet has a filter bank loss function
    and comes with functionality that tests the perfect
    reconstruction and anti-aliasing conditions.
    """

    @property
    @abstractmethod
    def filter_bank(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return dec_lo, dec_hi, rec_lo, rec_hi."""
        raise NotImplementedError

    @abstractmethod
    def wavelet_loss(self) -> torch.Tensor:
        """Return the sum of all loss terms."""
        return self.alias_cancellation_loss()[0] + self.perfect_reconstruction_loss()[0]

    @abstractmethod
    def __len__(self) -> int:
        """Return the filter length."""
        raise NotImplementedError

    # @abstractmethod
    # def parameters(self):
    #     raise NotImplementedError

    def pf_alias_cancellation_loss(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the product filter-alias cancellation loss.

        See: Strang+Nguyen 105: $$F_0(z) = H_1(-z); F_1(z) = -H_0(-z)$$
        Alternating sign convention from 0 to N see Strang overview
        on the back of the cover.

        Returns:
            The numerical value of the alias cancellation loss,
            as well as both loss components for analysis.
        """
        dec_lo, dec_hi, rec_lo, rec_hi = self.filter_bank
        m1 = torch.tensor([-1], device=dec_lo.device, dtype=dec_lo.dtype)
        length = dec_lo.shape[0]
        mask = torch.tensor(
            [torch.pow(m1, n) for n in range(length)][::-1],
            device=dec_lo.device,
            dtype=dec_lo.dtype,
        )
        err1 = rec_lo - mask * dec_hi
        err1s = torch.sum(err1 * err1)

        length = dec_lo.shape[0]
        mask = torch.tensor(
            [torch.pow(m1, n) for n in range(length)][::-1],
            device=dec_lo.device,
            dtype=dec_lo.dtype,
        )
        err2 = rec_hi - m1 * mask * dec_lo
        err2s = torch.sum(err2 * err2)
        return err1s + err2s, err1, err2

    def alias_cancellation_loss(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the alias cancellation loss.

        Implementation of the ac-loss as described
        on page 104 of Strang+Nguyen.
        $$F_0(z)H_0(-z) + F_1(z)H_1(-z) = 0$$

        Returns:
            The numerical value of the alias cancellation loss,
            as well as both loss components for analysis.
        """
        dec_lo, dec_hi, rec_lo, rec_hi = self.filter_bank
        m1 = torch.tensor([-1], device=dec_lo.device, dtype=dec_lo.dtype)
        length = dec_lo.shape[0]
        mask = torch.tensor(
            [torch.pow(m1, n) for n in range(length)][::-1],
            device=dec_lo.device,
            dtype=dec_lo.dtype,
        )
        # polynomial multiplication is convolution, compute p(z):
        pad = dec_lo.shape[0] - 1
        p_lo = torch.nn.functional.conv1d(
            dec_lo.unsqueeze(0).unsqueeze(0) * mask,
            torch.flip(rec_lo, [-1]).unsqueeze(0).unsqueeze(0),
            padding=pad,
        )

        pad = dec_hi.shape[0] - 1
        p_hi = torch.nn.functional.conv1d(
            dec_hi.unsqueeze(0).unsqueeze(0) * mask,
            torch.flip(rec_hi, [-1]).unsqueeze(0).unsqueeze(0),
            padding=pad,
        )

        p_test = p_lo + p_hi
        zeros = torch.zeros(p_test.shape, device=p_test.device, dtype=p_test.dtype)
        errs = (p_test - zeros) * (p_test - zeros)
        return torch.sum(errs), p_test, zeros

    def perfect_reconstruction_loss(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the perfect reconstruction loss.

        Returns:
            The numerical value of the alias cancellation loss,
            as well as both intermediate values for analysis.
        """
        # Strang 107: Assuming alias cancellation holds:
        # P(z) = F(z)H(z)
        # Product filter P(z) + P(-z) = 2.
        # However, since alias cancellation is implemented as a soft constraint:
        # P_0 + P_1 = 2
        # Somehow NumPy and PyTorch implement convolution differently.
        # For some reason, the machine learning people call cross-correlation
        # convolution.
        # https://discuss.pytorch.org/t/numpy-convolve-and-conv1d-in-pytorch/12172
        # Therefore for true convolution, one element needs to be flipped.

        dec_lo, dec_hi, rec_lo, rec_hi = self.filter_bank
        # polynomial multiplication is convolution, compute p(z):
        pad = dec_lo.shape[0] - 1
        p_lo = torch.nn.functional.conv1d(
            dec_lo.unsqueeze(0).unsqueeze(0),
            torch.flip(rec_lo, [-1]).unsqueeze(0).unsqueeze(0),
            padding=pad,
        )

        pad = dec_hi.shape[0] - 1
        p_hi = torch.nn.functional.conv1d(
            dec_hi.unsqueeze(0).unsqueeze(0),
            torch.flip(rec_hi, [-1]).unsqueeze(0).unsqueeze(0),
            padding=pad,
        )

        p_test = p_lo + p_hi
        two_at_power_zero = torch.zeros(
            p_test.shape, device=p_test.device, dtype=p_test.dtype
        )
        # numpy comparison for debugging.
        # np.convolve(self.init_wavelet.filter_bank[0],
        #             self.init_wavelet.filter_bank[2])
        # np.convolve(self.init_wavelet.filter_bank[1],
        #             self.init_wavelet.filter_bank[3])
        two_at_power_zero[..., p_test.shape[-1] // 2] = 2
        # square the error
        errs = (p_test - two_at_power_zero) * (p_test - two_at_power_zero)
        return torch.sum(errs), p_test, two_at_power_zero


class ProductFilter(WaveletFilter, torch.nn.Module):
    """Learnable product filter implementation."""

    def __init__(
        self,
        dec_lo: torch.Tensor,
        dec_hi: torch.Tensor,
        rec_lo: torch.Tensor,
        rec_hi: torch.Tensor,
    ):
        """Create a Product filter object.

        Args:
            dec_lo (torch.Tensor): Low pass analysis filter.
            dec_hi (torch.Tensor): High pass analysis filter.
            rec_lo (torch.Tensor): Low pass synthesis filter.
            rec_hi (torch.Tensor): High pass synthesis filter.
        """
        super().__init__()
        self.dec_lo = torch.nn.Parameter(dec_lo)
        self.dec_hi = torch.nn.Parameter(dec_hi)
        self.rec_lo = torch.nn.Parameter(rec_lo)
        self.rec_hi = torch.nn.Parameter(rec_hi)

    @property
    def filter_bank(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """All filters a a tuple."""
        return self.dec_lo, self.dec_hi, self.rec_lo, self.rec_hi

    # def parameters(self):
    #     return [self.dec_lo, self.dec_hi, self.rec_lo, self.rec_hi]

    def __len__(self) -> int:
        """Return the length of all filter arrays."""
        return self.dec_lo.shape[-1]

    def product_filter_loss(self) -> torch.Tensor:
        """Get only the product filter loss.

        Returns:
            The loss scalar.
        """
        return self.perfect_reconstruction_loss()[0] + self.alias_cancellation_loss()[0]

    def wavelet_loss(self) -> torch.Tensor:
        """Return the sum of all loss terms.

        Returns:
            The loss scalar.
        """
        return self.product_filter_loss()


class SoftOrthogonalWavelet(ProductFilter, torch.nn.Module):
    """Orthogonal wavelets with a soft orthogonality constraint."""

    def __init__(
        self,
        dec_lo: torch.Tensor,
        dec_hi: torch.Tensor,
        rec_lo: torch.Tensor,
        rec_hi: torch.Tensor,
    ):
        """Create a SoftOrthogonalWavelet object.

        Args:
            dec_lo (torch.Tensor): Low pass analysis filter.
            dec_hi (torch.Tensor): High pass analysis filter.
            rec_lo (torch.Tensor): Low pass synthesis filter.
            rec_hi (torch.Tensor): High pass synthesis filter.
        """
        super().__init__(dec_lo, dec_hi, rec_lo, rec_hi)

    def rec_lo_orthogonality_loss(self) -> torch.Tensor:
        """Return a Strang inspired soft orthogonality loss.

        See Strang p. 148/149 or Harbo p. 80.
        Since L is a convolution matrix, LL^T can be evaluated
        trough convolution.

        Returns:
            A tensor with the orthogonality constraint value.
        """
        filt_len = self.dec_lo.shape[-1]
        pad_dec_lo = torch.cat(
            [
                self.dec_lo,
                torch.zeros(
                    [
                        filt_len,
                    ],
                    device=self.dec_lo.device,
                ),
            ],
            -1,
        )
        res = torch.nn.functional.conv1d(
            pad_dec_lo.unsqueeze(0).unsqueeze(0),
            self.dec_lo.unsqueeze(0).unsqueeze(0),
            stride=2,
        )
        test = torch.zeros_like(res.squeeze(0).squeeze(0))
        test[0] = 1
        err = res - test
        return torch.sum(err * err)

    def filt_bank_orthogonality_loss(self) -> torch.Tensor:
        """Return a Jensen+Harbo inspired soft orthogonality loss.

        On Page 79 of the Book Ripples in Mathematics
        by Jensen la Cour-Harbo, the constraint
        g0[k] = h0[-k] and g1[k] = h1[-k] for orthogonal filters
        is presented. A measurement is implemented below.

        Returns:
            A tensor with the orthogonality constraint value.
        """
        eq0 = self.dec_lo - self.rec_lo.flip(-1)
        eq1 = self.dec_hi - self.rec_hi.flip(-1)
        seq0 = torch.sum(eq0 * eq0)
        seq1 = torch.sum(eq1 * eq1)
        # print(eq0, eq1)
        return seq0 + seq1

    def wavelet_loss(self) -> torch.Tensor:
        """Return the sum of all terms."""
        return self.product_filter_loss() + self.filt_bank_orthogonality_loss()
