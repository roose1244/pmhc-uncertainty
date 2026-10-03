import unittest

from src.hla_seq import apply_engineered, mature_groove, parse_imgt_fasta

GROOVE = "GSHSM" + "A" * 177
FASTA = (
    ">HLA:HLA00005 A*02:01:01:01 365 bp\n"
    + ("M" * 24 + GROOVE + "X" * 20)
    + "\n"
)


class HlaSeqTests(unittest.TestCase):
    def test_two_field_and_groove(self):
        lookup = parse_imgt_fasta(FASTA)
        self.assertIn("HLA-A*02:01", lookup)
        groove = mature_groove(lookup["HLA-A*02:01"])
        self.assertEqual(len(groove), 182)
        self.assertTrue(groove.startswith("GSHSM"))

    def test_c67s(self):
        groove = "C" * 182
        mutated = apply_engineered("HLA-B*14:02(C67S)", groove)
        self.assertEqual(mutated[66], "S")
        self.assertEqual(mutated[65], "C")
        self.assertEqual(apply_engineered("HLA-B*14:02", groove)[66], "C")


if __name__ == "__main__":
    unittest.main()
