import unittest

import numpy as np

from isotools_extensions.psi_test import (
    gene_coverage_sum_samplewise,
    gene_coverage_sum_transcriptwise,
)


class TestGeneCoverageSumSamplewise(unittest.TestCase):
    def setUp(self):
        """Set up common data objects."""
        self.coverage = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
            ],
        )
        self.sample_groups = {
            "group1": [0, 1],
            "group2": [2, 3],
        }

    def test_basic_functionality(self):
        """Test basic functionality."""
        expected_counts = np.array(
            [
                [5, 7, 9],  # Sum of rows 0 and 1
                [17, 19, 21],  # Sum of rows 2 and 3
            ],
        )
        expected_keys = ["group1", "group2"]

        counts, keys = gene_coverage_sum_samplewise(self.coverage, self.sample_groups)
        np.testing.assert_array_equal(counts, expected_counts)
        self.assertEqual(keys, expected_keys)

    def test_descending_order(self):
        """Test with samples sorted in descending order."""
        expected_counts_desc = np.array(
            [
                [17, 19, 21],  # Sum of rows 2 and 3
                [5, 7, 9],  # Sum of rows 0 and 1
            ],
        )
        expected_keys_desc = ["group2", "group1"]

        counts_desc, keys_desc = gene_coverage_sum_samplewise(
            self.coverage,
            self.sample_groups,
            sort_descending=True,
        )
        np.testing.assert_array_equal(counts_desc, expected_counts_desc)
        self.assertEqual(keys_desc, expected_keys_desc)

    def test_empty_sample_groups(self):
        """Test with empty group."""
        sample_groups_empty = {}
        counts_empty, keys_empty = gene_coverage_sum_samplewise(
            self.coverage,
            sample_groups_empty,
        )
        self.assertEqual(np.size(counts_empty), 0)
        self.assertEqual(np.size(keys_empty), 0)

    def test_single_group_all_samples(self):
        """Test with single group that contains all samples."""
        sample_groups_all = {"all": [0, 1, 2, 3]}
        expected_counts_all = np.array(
            [
                [22, 26, 30],  # Sum of all rows
            ],
        )
        expected_keys_all = ["all"]

        counts_all, keys_all = gene_coverage_sum_samplewise(
            self.coverage,
            sample_groups_all,
        )
        np.testing.assert_array_equal(counts_all, expected_counts_all)
        self.assertEqual(keys_all, expected_keys_all)

    def test_non_overlapping_groups(self):
        """Test with non-overlapping groups."""
        sample_groups_nonoverlap = {
            "group1": [0],
            "group2": [1],
            "group3": [2],
            "group4": [3],
        }
        expected_counts_nonoverlap = np.array(
            [
                [1, 2, 3],  # Row 0
                [4, 5, 6],  # Row 1
                [7, 8, 9],  # Row 2
                [10, 11, 12],  # Row 3
            ],
        )
        expected_keys_nonoverlap = ["group1", "group2", "group3", "group4"]

        counts_nonoverlap, keys_nonoverlap = gene_coverage_sum_samplewise(
            self.coverage,
            sample_groups_nonoverlap,
        )
        np.testing.assert_array_equal(counts_nonoverlap, expected_counts_nonoverlap)
        self.assertEqual(keys_nonoverlap, expected_keys_nonoverlap)

    def test_invalid_sample_index(self):
        """Test with invalid sample index."""
        sample_groups_invalid = {
            "group1": [0, 1],
            "group2": [2, 4],  # Index 4 is out of bounds
        }
        with self.assertRaises(IndexError):
            gene_coverage_sum_samplewise(self.coverage, sample_groups_invalid)


class TestGeneCoverageSumTranscriptwise(unittest.TestCase):
    def setUp(self):
        """Set up common data for the tests."""
        self.coverage = np.array(
            [
                [1, 2, 3, 4],  # Sample 1
                [5, 6, 7, 8],  # Sample 2
                [9, 10, 11, 12],  # Sample 3
            ],
        )
        self.transcript_groups = {
            "group1": [0, 1],  # Transcripts 0 and 1
            "group2": [2, 3],  # Transcripts 2 and 3
        }

    def test_basic_functionality(self):
        """Test basic functionality with ascending order."""
        expected_counts = np.array(
            [
                [3, 7],  # Sample 1: group1 = 1+2, group2 = 3+4
                [11, 15],  # Sample 2: group1 = 5+6, group2 = 7+8
                [19, 23],  # Sample 3: group1 = 9+10, group2 = 11+12
            ],
        )
        expected_keys = ["group1", "group2"]
        counts, keys = gene_coverage_sum_transcriptwise(
            self.coverage,
            self.transcript_groups,
        )
        np.testing.assert_array_equal(counts, expected_counts)
        self.assertEqual(keys, expected_keys)

    def test_sort_descending(self):
        """Test functionality with descending order."""
        expected_counts = np.array(
            [
                [7, 3],  # Sample 1: group2 = 3+4, group1 = 1+2
                [15, 11],  # Sample 2: group2 = 7+8, group1 = 5+6
                [23, 19],  # Sample 3: group2 = 11+12, group1 = 9+10
            ],
        )
        expected_keys = ["group2", "group1"]
        counts, keys = gene_coverage_sum_transcriptwise(
            self.coverage,
            self.transcript_groups,
            sort_descending=True,
        )
        np.testing.assert_array_equal(counts, expected_counts)
        self.assertEqual(keys, expected_keys)

    def test_empty_transcript_groups(self):
        """Test with empty transcript_groups."""
        empty_transcript_groups = {}
        counts, keys = gene_coverage_sum_transcriptwise(
            self.coverage,
            empty_transcript_groups,
        )
        self.assertEqual(np.size(counts), 0)
        self.assertEqual(np.size(keys), 0)

    def test_empty_coverage(self):
        """Test with empty coverage array."""
        empty_coverage = np.empty((0, 4))  # 0 samples, 4 transcripts
        expected_counts = np.empty((0, 2))  # 0 samples, 2 groups
        expected_keys = ["group1", "group2"]
        counts, keys = gene_coverage_sum_transcriptwise(
            empty_coverage,
            self.transcript_groups,
        )
        np.testing.assert_array_equal(counts, expected_counts)
        self.assertEqual(keys, expected_keys)

    def test_single_transcript_group(self):
        """Test with a single transcript group."""
        single_transcript_group = {
            "group1": [0, 1, 2, 3],
        }  # All transcripts in one group
        expected_counts = np.array(
            [
                [10],  # Sample 1: group1 = 1+2+3+4
                [26],  # Sample 2: group1 = 5+6+7+8
                [42],  # Sample 3: group1 = 9+10+11+12
            ],
        )
        expected_keys = ["group1"]
        counts, keys = gene_coverage_sum_transcriptwise(
            self.coverage,
            single_transcript_group,
        )
        np.testing.assert_array_equal(counts, expected_counts)
        self.assertEqual(keys, expected_keys)

    def test_non_contiguous_transcripts(self):
        """Test with non-contiguous transcript indices."""
        non_contiguous_groups = {
            "group1": [0, 2],  # Transcripts 0 and 2
            "group2": [1, 3],  # Transcripts 1 and 3
        }
        expected_counts = np.array(
            [
                [4, 6],  # Sample 1: group1 = 1+3, group2 = 2+4
                [12, 14],  # Sample 2: group1 = 5+7, group2 = 6+8
                [20, 22],  # Sample 3: group1 = 9+11, group2 = 10+12
            ],
        )
        expected_keys = ["group1", "group2"]
        counts, keys = gene_coverage_sum_transcriptwise(
            self.coverage,
            non_contiguous_groups,
        )
        np.testing.assert_array_equal(counts, expected_counts)
        self.assertEqual(keys, expected_keys)


if __name__ == "__main__":
    unittest.main()
