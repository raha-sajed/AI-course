import unittest

import numpy as np

from HomeWork1.signal_analysis import extract_features, segment_signal


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
        result = extract_features(windows, starts)
        # RMS^2 = variance + mean^2 for population variance.
        np.testing.assert_allclose(result.rms ** 2, result["std"] ** 2 + result["mean"] ** 2)

    def test_invalid_overlap_and_nonfinite_signal_are_rejected(self):
        with self.assertRaises(ValueError):
            segment_signal(np.ones(20), 4, 4)
        with self.assertRaises(ValueError):
            segment_signal(np.array([1, 2, np.nan, 4]), 4, 0)


if __name__ == "__main__":
    unittest.main()
