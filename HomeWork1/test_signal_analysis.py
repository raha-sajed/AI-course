import unittest

import numpy as np

from HomeWork1.signal_analysis import compute_spectrum, extract_features, segment_signal, signal_statistics, time_window


class SignalFeatureTests(unittest.TestCase):
    def test_sample_positions_survive_overlap_and_incomplete_tail(self):
        signal = np.arange(11, dtype=float)
        windows, starts = segment_signal(signal, length=4, overlap=2)
        np.testing.assert_array_equal(starts, [0, 2, 4, 6])
        reconstructed = np.full(10, np.nan)
        for window, start in zip(windows, starts):
            reconstructed[start:start + 4] = window
        np.testing.assert_array_equal(reconstructed, signal[:10])

    def test_energy_identity_for_each_segment(self):
        windows, starts = segment_signal(np.array([0, 3, -4, 2, 5, 1, 9, -2]), 4, 2)
        result = extract_features(windows)
        # RMS^2 = variance + mean^2 for population variance.
        np.testing.assert_allclose(result.rms ** 2, result["std"] ** 2 + result["mean"] ** 2)

    def test_spectrum_recovers_known_frequency_and_amplitude_with_dc_offset(self):
        fs = 1000.
        for n in (1000, 1001):
            frequency = 50 * fs / n
            signal = 7 + 3 * np.sin(2 * np.pi * frequency * np.arange(n) / fs)
            bins, amplitude, peak = compute_spectrum(signal, fs)
            self.assertAlmostEqual(peak, frequency, places=8)
            self.assertAlmostEqual(amplitude[50], 3., places=8)
            self.assertAlmostEqual(amplitude[0], 0., places=8)
        _, _, peak = compute_spectrum(np.full(100, 7.), fs)
        self.assertIsNone(peak)

    def test_nyquist_amplitude_is_not_doubled(self):
        _, amplitude, peak = compute_spectrum(2. * (-1.) ** np.arange(1000), 1000.)
        self.assertAlmostEqual(amplitude[-1], 2.)
        self.assertEqual(peak, 500.)

    def test_zoom_preserves_original_sample_positions(self):
        signal = np.arange(150000)
        times, values = time_window(signal, 48000.)
        self.assertEqual(len(values), 480)
        np.testing.assert_array_equal(values, signal[96000:96480])
        self.assertEqual(times[0], 2.)
        self.assertLess(times[-1], 2.01)

    def test_statistics_and_export_schema_match_assignment(self):
        np.testing.assert_allclose(signal_statistics([3, 3]), [3, 0, 3])
        features = extract_features(np.array([[3., 3.], [-2., 2.]]))
        self.assertEqual(list(features.columns), ["mean", "std", "rms"])
        np.testing.assert_allclose(features.iloc[1], [0, 2, 2])

    def test_invalid_overlap_and_nonfinite_signal_are_rejected(self):
        with self.assertRaises(ValueError):
            segment_signal(np.ones(20), 4, 4)
        with self.assertRaises(ValueError):
            segment_signal(np.array([1, 2, np.nan, 4]), 4, 0)


if __name__ == "__main__":
    unittest.main()
