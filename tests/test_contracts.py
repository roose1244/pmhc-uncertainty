import unittest

from src.hla import normalise_hla
from src.target import from_log, make_id, to_log


class HlaTests(unittest.TestCase):
    def test_already_canonical(self):
        self.assertEqual(normalise_hla("HLA-A*02:01"), "HLA-A*02:01")

    def test_short_and_compact(self):
        self.assertEqual(normalise_hla("A*02:01"), "HLA-A*02:01")
        self.assertEqual(normalise_hla("A0201"), "HLA-A*02:01")
        self.assertEqual(normalise_hla("hla-b*15:02"), "HLA-B*15:02")

    def test_engineered_mutant(self):
        self.assertEqual(
            normalise_hla("HLA-B*14:02(C67S)"),
            "HLA-B*14:02(C67S)",
        )
        self.assertEqual(normalise_hla("B*14:02(C67S)"), "HLA-B*14:02(C67S)")

    def test_rejects_garbage(self):
        with self.assertRaises(ValueError):
            normalise_hla("mouse-H2-Kb")


class TargetTests(unittest.TestCase):
    def test_zero_is_defined(self):
        self.assertAlmostEqual(to_log(0.0), -1.0)

    def test_roundtrip_positive(self):
        self.assertAlmostEqual(from_log(to_log(8.4)), 8.4)

    def test_id_joins_normalised_allele(self):
        self.assertEqual(make_id("A0201", "vttevafgl"), "HLA-A*02:01|VTTEVAFGL")


if __name__ == "__main__":
    unittest.main()
