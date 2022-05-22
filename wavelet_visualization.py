"""
Basic wavelet visualization.
"""

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import chirp, gausspulse


def make_chirp(
    sampling_freq: int, start_chirp_freq: float, end_chirp_freq: float
) -> np.ndarray:
    """
    Make linear FM chirp, ramping from start_chirp_freq to end_chirp_freq.

    Args:
        sampling_freq: Sampling frequency (Hz).
        start_chirp_freq: Start frequency (Hz).
        end_chirp_freq: End frequency (Hz).
    Returns:
        Linear FM chirp.
    """

    return chirp(np.linspace(0, 2, sampling_freq), start_chirp_freq, 2, end_chirp_freq)


def zero_padded_wavelet(
    offset: float, center_freq: float, sampling_freq: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shift wavelet in time.

    Args:
        offset: Offset to apply to wavelet.
        center_freq: Wavelet frequency (Hz).
        sampling_freq: Sampling frequency (Hz).
    Returns:
        Wavelet offset in time.
    """

    xs = np.linspace(  # pylint: disable=invalid-name
        -1 - offset, 1 - offset, sampling_freq
    )
    ys_real, ys_imag = gausspulse(xs, fc=center_freq, retquad=True)

    return ys_real, ys_imag


def plot_chirp(
    ax: plt.Axes, freqs: np.ndarray, chirp_: np.ndarray  # pylint: disable=invalid-name
) -> None:
    """
    Visualize chirp.

    Args:
        ax: Matplotlib axis.
        frequencies: Chirp frequencies.
        chirp: Linear FM chirp.
    """

    plt.sca(ax)
    plt.plot(freqs, chirp_)
    freq_min, freq_max = np.min(freqs), np.max(freqs)
    plt.xlim(freq_min, freq_max)
    plt.xticks(np.arange(freq_min, freq_max, 1))
    plt.yticks([])


def plot_wavelet(
    ax: plt.Axes,  # pylint: disable=invalid-name
    wavelet_real: np.ndarray,
    wavelet_imag: np.ndarray,
    ylim: Tuple[float, float] = (-2.0, 2.0),
) -> None:
    """
    Visualize wavelet.

    Args:
        ax: Matplotlib axis.
        wavelet_real: Real wavelet component.
        wavelet_imag: Imag wavelet componenet.
    """

    plt.sca(ax)
    plt.plot(wavelet_real)
    plt.plot(wavelet_imag)
    plt.ylim(*ylim)
    plt.xlim(0, len(wavelet_real))
    plt.xticks([])
    plt.yticks([])


def plot_wavelet_response(
    ax: plt.Axes,  # pylint: disable=invalid-name
    chirp_: np.ndarray,
    wavelet_real: np.ndarray,
    wavelet_imag: np.ndarray,
    ylim: Tuple[float, float] = (-2.0, 2.0),
) -> None:
    """
    Visualize pointwise multiplication between wavelet and chirp.

    Args:
        ax: Matplotlib axis.
        chirp: Linear FM chirp.
        wavelet_real: Real wavelet component.
        wavelet_imag: Imag wavelet componenet.
    """

    plt.sca(ax)
    plt.plot(chirp_ * wavelet_real)
    plt.plot(chirp_ * wavelet_imag)
    plt.xlim(0, len(wavelet_real))
    plt.ylim(*ylim)
    plt.xticks([])
    plt.yticks([])


def plot_wavelet_convolution(  # pylint: disable=too-many-arguments
    ax: plt.Axes,  # pylint: disable=invalid-name
    freqs: np.ndarray,
    chirp_: np.ndarray,
    wavelet_real: np.ndarray,
    wavelet_imag: np.ndarray,
    sums: List[Tuple],
    ylim: Tuple[float, float] = (0.0, 150.0),
) -> None:
    """
    Visualize convolution values.

    Args:
        ax: Matplotlib axis.
        chirp: Linear FM chirp.
        wavelet_real: Real wavelet component.
        wavelet_imag: Imag wavelet componenet.
        sums: Convolution values.
    """

    plt.sca(ax)

    real_kernel_center = np.argmax(wavelet_real)

    # magnitude of complex value
    sum_ = np.sum(chirp_ * wavelet_real) ** 2 + np.sum(chirp_ * wavelet_imag) ** 2

    if np.isclose(wavelet_real[real_kernel_center], 1, 0.1):
        kernel_center = real_kernel_center
        sums.append((freqs[kernel_center], sum_))

    if sums:
        wavelet_sum_xs, wavelet_sum_ys = zip(*sums)
        plt.plot(wavelet_sum_xs, wavelet_sum_ys)

    freq_min, freq_max = np.min(freqs), np.max(freqs)
    plt.xlim(freq_min, freq_max)
    plt.xticks(np.arange(freq_min, freq_max, 1))
    plt.ylim(*ylim)
    plt.yticks([])


def wavelet_demo():
    """ Wavelet visualization for Medium article. """

    sampling_hz = 200
    wavelet_freq = 8
    start_chirp_freq = 1.0
    end_chirp_freq = 15.0

    chirp_ = make_chirp(sampling_hz, start_chirp_freq, end_chirp_freq)
    freqs = np.linspace(start_chirp_freq, end_chirp_freq, sampling_hz)

    plt.figure()
    plt.ion()

    sums: List[Tuple] = []

    for wavelet_offset in np.arange(-1.25, 1.25, 0.01):

        plt.clf()

        wavelet_real, wavelet_imag = zero_padded_wavelet(
            wavelet_offset, wavelet_freq, sampling_hz
        )

        ax1 = plt.subplot(4, 1, 1)
        plot_chirp(ax1, freqs, chirp_)

        ax2 = plt.subplot(4, 1, 2)
        plot_wavelet(ax2, wavelet_real, wavelet_imag)

        ax3 = plt.subplot(4, 1, 3)
        plot_wavelet_response(ax3, chirp_, wavelet_real, wavelet_imag)

        ax4 = plt.subplot(4, 1, 4)
        plot_wavelet_convolution(ax4, freqs, chirp_, wavelet_real, wavelet_imag, sums)

        plt.show()
        plt.pause(0.001)


if __name__ == "__main__":
    wavelet_demo()
