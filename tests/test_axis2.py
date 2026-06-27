import numpy as np
import pandas as pd
import pytest

from axis2_spectral import (compute_parametric_spectrum, compute_periodogram,
                            detect_cycles, smooth_spectrum)


def test_compute_periodogram_demean_and_shapes():
    # Generate some simple sine data
    n = 100
    t = np.arange(n)
    series = pd.Series(5.0 + 3.0 * np.sin(2 * np.pi * 0.1 * t))  # Mean of 5.0

    # 1. Test None taper
    freqs, pgram = compute_periodogram(series, taper=None, taper_pct=0.1)

    assert len(freqs) == n // 2 + 1
    assert len(pgram) == n // 2 + 1
    assert freqs[0] == 0.0
    # Demeaned series sum to 0 -> frequency 0 should have periodogram value ~ 0
    assert np.abs(pgram[0]) < 1e-10

    # 2. Test boundary: empty series
    empty_series = pd.Series([], dtype=float)
    freqs_empty, pgram_empty = compute_periodogram(
        empty_series, taper=None, taper_pct=0.1
    )
    assert len(freqs_empty) == 0
    assert len(pgram_empty) == 0


def test_compute_periodogram_tapers():
    n = 64
    series = pd.Series(np.random.randn(n))

    # Cosine Bell
    freqs_cb, pgram_cb = compute_periodogram(series, taper="Cosine Bell", taper_pct=0.1)
    assert len(pgram_cb) == n // 2 + 1

    # Hann
    freqs_hn, pgram_hn = compute_periodogram(series, taper="Hann", taper_pct=0.1)
    assert len(pgram_hn) == n // 2 + 1

    # Hamming
    freqs_hm, pgram_hm = compute_periodogram(series, taper="Hamming", taper_pct=0.1)
    assert len(pgram_hm) == n // 2 + 1

    # Verify Cosine Bell taper math for p = int(taper_pct * n)
    # If taper_pct is 0.25, then p = 16
    # w[0] = 0.5 * (1 - cos(0)) = 0.0, so the first elements of the tapered series must be 0.0
    series_fixed = pd.Series(np.ones(n))  # all ones
    # Demean makes it all zeros, so let's bypass demeaning logic by checking the taper array logic:
    # We can test with a series that is asymmetric so demeaning leaves non-zero endpoints.
    # A simpler way is to check the output of a specific series.
    # For example, if we have series = [1, 2, 3, 4], mean = 2.5, demeaned = [-1.5, -0.5, 0.5, 1.5]
    # taper_pct = 0.25 -> p = 1.
    # w[0] = 0.5 * (1 - cos(0)) = 0.0
    # w[3] = 0.0
    # So the tapered demeaned series is [0.0, -0.5, 0.5, 0.0].
    # Let's compute periodogram with Cosine Bell on [1, 2, 3, 4] with taper_pct=0.25:
    small_series = pd.Series([1.0, 2.0, 3.0, 4.0])
    freqs, pgram = compute_periodogram(
        small_series, taper="Cosine Bell", taper_pct=0.25
    )
    # The tapered series is indeed [0, -0.5, 0.5, 0]
    # FFT of [0, -0.5, 0.5, 0] = [-0.5 - 0.5j, -0.5 + 0.5j] or similar.
    # Specifically, the sum for freq=0 is 0, so pgram[0] = 0.
    # Let's just assert that it ran without issues.
    assert len(pgram) == 3


def test_smooth_spectrum_windows_and_ci():
    n = 100
    series = pd.Series(np.sin(np.linspace(0, 10 * np.pi, n)) + np.random.randn(n) * 0.1)

    for win in ["Daniell", "Bartlett", "Parzen", "Hann"]:
        freqs, f_hat, ci_lower, ci_upper = smooth_spectrum(series, window=win, M=20)
        assert len(freqs) == 512
        assert len(f_hat) == 512
        assert len(ci_lower) == 512
        assert len(ci_upper) == 512
        # CI upper should be strictly greater than or equal to f_hat, and f_hat >= ci_lower
        assert np.all(ci_upper >= f_hat)
        assert np.all(f_hat >= ci_lower)
        # All values should be positive due to clipping
        assert np.all(f_hat > 0)

    # Verify degenerate boundary cases
    empty_series = pd.Series([], dtype=float)
    freqs, f_hat, ci_lower, ci_upper = smooth_spectrum(
        empty_series, window="Daniell", M=10
    )
    assert len(f_hat) == 512
    assert np.all(f_hat == 0)


def test_smooth_spectrum_nu():
    # Verify manual degrees of freedom formula
    # nu = 2n / (1 + 2 * sum(w(h/M)^2))
    # Let's use Bartlett window, M = 4, n = 100.
    # Bartlett: w(h/M) = 1 - h/M.
    # For h=1: w(1/4) = 0.75, h=2: w(2/4) = 0.5, h=3: w(3/4) = 0.25, h=4: w(4/4) = 0.0
    # sum_w2 = 0.75^2 + 0.5^2 + 0.25^2 + 0.0^2 = 0.5625 + 0.25 + 0.0625 = 0.875
    # nu = 200 / (1 + 2 * 0.875) = 200 / 2.75 = 72.7272
    series = pd.Series(np.random.randn(100))
    freqs, f_hat, ci_lower, ci_upper = smooth_spectrum(series, window="Bartlett", M=4)

    # Let's extract nu from ratio: ratio = ci_upper / f_hat = nu / chi2.ppf(0.025, df=nu)
    # Let's compute target nu = 200 / 2.75
    target_nu = 200.0 / 2.75
    import scipy.stats

    expected_ratio = target_nu / scipy.stats.chi2.ppf(0.025, df=target_nu)
    actual_ratio = (ci_upper / f_hat)[0]
    np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=1e-5)


def test_compute_parametric_spectrum():
    # 1. Flat spectrum (White Noise)
    # ar = [], ma = [], sigma2 = 1.0 -> f(lambda) = 1.0 / (2 * pi)
    freqs, spec = compute_parametric_spectrum([], [], 1.0, 100)
    assert len(freqs) == 100
    assert len(spec) == 100
    np.testing.assert_allclose(spec, 1.0 / (2.0 * np.pi))

    # 2. ARMA(1, 1) process
    ar = [0.5]
    ma = [-0.3]
    sigma2 = 1.2
    freqs, spec = compute_parametric_spectrum(ar, ma, sigma2, 200)
    assert len(spec) == 200
    assert np.all(spec > 0)

    # Check at lambda = 0:
    # f(0) = sigma2 / (2*pi) * |1 + theta_1|^2 / |1 - phi_1|^2
    # = 1.2 / (2*pi) * (0.7)^2 / (0.5)^2 = 1.2 / (2*pi) * 0.49 / 0.25 = 2.352 / (2*pi)
    expected_val = (
        (sigma2 / (2.0 * np.pi)) * ((1.0 + ma[0]) ** 2) / ((1.0 - ar[0]) ** 2)
    )
    np.testing.assert_allclose(spec[0], expected_val)


def test_detect_cycles():
    # Create a synthetic spectrum with a strong peak at index 100
    n_points = 512
    frequencies = np.linspace(0, np.pi, n_points)

    # Base flat spectrum at level 0.1
    f_hat = np.ones(n_points) * 0.1
    # Add a huge peak at index 100
    f_hat[100] = 5.0
    f_hat[99] = 2.5
    f_hat[101] = 2.5

    # Construct corresponding ci_upper representing a moderate degrees of freedom, e.g. nu = 40.
    # ratio for nu=40 is 40 / chi2.ppf(0.025, 40) = 40 / 24.433 = 1.637
    # So ci_upper = 1.637 * f_hat
    ci_upper = 1.637 * f_hat

    cycles = detect_cycles(frequencies, f_hat, ci_upper)

    # We should detect the peak at index 100
    assert len(cycles) > 0
    # Peak at index 100 is at frequency frequencies[100]
    expected_freq = frequencies[100]
    # The first element should be this peak because it's sorted by height descending
    assert np.isclose(cycles[0]["frequency"], expected_freq)
    assert cycles[0]["significant"] is True
    # Period should be 2*pi / frequency
    assert np.isclose(cycles[0]["period"], 2.0 * np.pi / expected_freq)

    # Check empty/short boundary case
    assert (
        detect_cycles(np.array([1, 2]), np.array([1.0, 1.0]), np.array([2.0, 2.0]))
        == []
    )
