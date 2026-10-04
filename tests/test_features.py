import unittest

import numpy as np

from src.features import blosum_chain, blosum_peptide, encode_hla, encode_peptides, fit_hla_encoder


class FeatureTests(unittest.TestCase):
    def test_blosum_shape(self):
        vec = blosum_peptide("GILGFVFTL")
        self.assertEqual(vec.shape, (180,))
        self.assertEqual(encode_peptides(["GILGFVFTL", "VTTEVAFGL"]).shape, (2, 180))

    def test_pocket_encoding_sees_contact_differences(self):
        same = blosum_chain("A" * 34)
        other = blosum_chain("A" * 33 + "C")
        self.assertEqual(same.shape, (34 * 20,))
        self.assertFalse((same == other).all())

    def test_unknown_hla_is_zero(self):
        encoder = fit_hla_encoder(["HLA-A*02:01", "HLA-B*07:02"])
        known = encode_hla(encoder, ["HLA-A*02:01"])
        unknown = encode_hla(encoder, ["HLA-B*15:02"])
        self.assertEqual(known.shape[1], 2)
        self.assertAlmostEqual(float(known.sum()), 1.0)
        self.assertAlmostEqual(float(unknown.sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
