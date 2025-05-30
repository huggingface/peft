"""Test the adaptive wavelet cost functions."""

import pytest
import pywt
import torch

from ptwt.wavelets_learnable import SoftOrthogonalWavelet


@pytest.mark.parametrize(
    "lst, is_orth",
    [
        (pywt.wavelist(family="db"), True),
        (pywt.wavelist(family="sym"), True),
        (pywt.wavelist(family="coif"), True),
        (pywt.wavelist(family="bior"), False),
        (pywt.wavelist(family="rbio"), False),
    ],
)
def test_wavelet_lst(lst: list[str], is_orth: bool) -> None:
    """Test all wavelets in a list."""
    for ws in lst:
        wavelet = pywt.Wavelet(ws)
        orthwave = SoftOrthogonalWavelet(
            torch.tensor(wavelet.dec_lo),
            torch.tensor(wavelet.dec_hi),
            torch.tensor(wavelet.rec_lo),
            torch.tensor(wavelet.rec_hi),
        )
        prl = orthwave.perfect_reconstruction_loss()[0]
        acl = orthwave.alias_cancellation_loss()[0]
        assert prl < 1e-10
        assert acl < 1e-10
        pacl = orthwave.pf_alias_cancellation_loss()[0]

        orth = orthwave.filt_bank_orthogonality_loss()
        print(
            ws,
            "prl, %.5f | acl, %.5f | pfacl, %.5f | orth, %.5f "
            % (prl.item(), acl.item(), pacl.item(), orth.item()),
        )
        if is_orth is True:
            assert orth < 1e-10
