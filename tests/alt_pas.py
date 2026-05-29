import unittest
import numpy as np

from isotools_extensions.alt_pas import (
    assign_to_closest_peak,
    translate_peaks_offset,
    mean_countsdict,
)


class TestAssignToClosestPeak(unittest.TestCase):
    def test_single_peak(self):
        """Test with a single peak in the list."""
        pos = 10
        peaks = [15]
        result = assign_to_closest_peak(pos, peaks)
        self.assertEqual(result, (0, 5))  # Closest peak is at index 0, distance is 5

    def test_multiple_peaks(self):
        """Test with multiple peaks in the list."""
        pos = 10
        peaks = [2, 15, 20]
        result = assign_to_closest_peak(pos, peaks)
        self.assertEqual(result, (1, 5))  # Closest peak is at index 1, distance is 5

    def test_exact_match(self):
        """Test when the position matches a peak exactly."""
        pos = 15
        peaks = [5, 15, 20]
        result = assign_to_closest_peak(pos, peaks)
        self.assertEqual(result, (1, 0))  # Closest peak is at index 1, distance is 0

    def test_negative_coordinates(self):
        """Test with negative genomic coordinates."""
        pos = -10
        peaks = [-20, -5, 0]
        result = assign_to_closest_peak(pos, peaks)
        self.assertEqual(result, (1, 5))  # Closest peak is at index 0, distance is 10

    def test_empty_peaks(self):
        """Test with an empty list of peaks."""
        pos = 10
        peaks = []
        with self.assertRaises(
            ValueError
        ):  # np.argmin raises ValueError for empty list
            assign_to_closest_peak(pos, peaks)

    def test_large_numbers(self):
        """Test with large genomic coordinates."""
        pos = 1_000_000
        peaks = [500_000, 1_700_000, 2_000_000]
        result = assign_to_closest_peak(pos, peaks)
        self.assertEqual(
            result, (0, 500_000)
        )  # Closest peak is at index 0, distance is 500,000


class TestTranslatePeaksOffset(unittest.TestCase):
    def test_basic(self):
        """Test the translate_peaks_offset function."""
        peaks = (
            [10, 20, 30],
            {
                "prominences": [0.5, 1.0, 1.5],
                "left_bases": [5, 15, 25],
                "right_bases": [15, 25, 35],
            },
        )
        offset = 100
        result = translate_peaks_offset(peaks, offset)
        expected_peaks = np.array([110, 120, 130])
        expected_dict = {
            "prominences": [0.5, 1.0, 1.5],
            "left_bases": np.array([105, 115, 125]),
            "right_bases": np.array([115, 125, 135]),
        }
        np.testing.assert_array_equal(result[0], expected_peaks)
        self.assertEqual(result[1]["prominences"], expected_dict["prominences"])
        np.testing.assert_array_equal(
            result[1]["left_bases"], expected_dict["left_bases"]
        )
        np.testing.assert_array_equal(
            result[1]["right_bases"], expected_dict["right_bases"]
        )

    def test_empty(self):
        """Test translate_peaks_offset with empty peaks."""
        peaks = (
            [],
            {
                "prominences": [],
                "left_bases": [],
                "right_bases": [],
            },
        )
        offset = 50
        result = translate_peaks_offset(peaks, offset)
        expected_peaks = np.array([])
        expected_dict = {
            "prominences": [],
            "left_bases": np.array([]),
            "right_bases": np.array([]),
        }
        np.testing.assert_array_equal(result[0], expected_peaks)
        self.assertEqual(result[1]["prominences"], expected_dict["prominences"])
        np.testing.assert_array_equal(
            result[1]["left_bases"], expected_dict["left_bases"]
        )
        np.testing.assert_array_equal(
            result[1]["right_bases"], expected_dict["right_bases"]
        )


class TestMeanCountsdict(unittest.TestCase):
    def test_basic(self):
        """Test the mean_countsdict function."""
        countsdict = {1: 2, 2: 3, 3: 5}
        result = mean_countsdict(countsdict)
        expected_mean = (1 * 2 + 2 * 3 + 3 * 5) / (2 + 3 + 5)  # Weighted mean
        expected_sum = 2 + 3 + 5  # Total count
        self.assertAlmostEqual(result[0], expected_mean)
        self.assertEqual(result[1], expected_sum)

    def test_empty(self):
        """Test mean_countsdict with an empty dictionary."""
        countsdict = {}
        with self.assertRaises(
            ZeroDivisionError
        ):  # Division by zero when calculating mean
            mean_countsdict(countsdict)

    def test_single_entry(self):
        """Test mean_countsdict with a single entry."""
        countsdict = {5: 10}
        result = mean_countsdict(countsdict)
        expected_mean = 5  # Only one key, so mean is the key itself
        expected_sum = 10  # Total count is the value
        self.assertEqual(result, (expected_mean, expected_sum))


# Run the tests
if __name__ == "__main__":
    unittest.main()
