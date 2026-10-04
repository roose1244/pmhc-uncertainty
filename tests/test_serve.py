import unittest

from src.serve import _verdict


class VerdictTests(unittest.TestCase):
    def test_unseen_allele_is_not_called_calibrated(self):
        text = _verdict(False, 0.1, 0.5, True)
        self.assertIn("Unseen HLA", text)
        self.assertIn("not a warning", text)

    def test_seen_allele_uses_the_calibration_cut(self):
        low = _verdict(True, 0.2, 0.5, True)
        high = _verdict(True, 0.8, 0.5, False)
        self.assertIn("lower half", low)
        self.assertIn("Peptide was not in training", low)
        self.assertIn("upper half", high)
        self.assertIn("Peptide was in training", high)


if __name__ == "__main__":
    unittest.main()
