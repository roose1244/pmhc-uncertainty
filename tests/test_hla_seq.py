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

    def test_three_digit_protein_is_not_the_two_digit_allele(self):
        text = (
            ">HLA:HLA00017 A*02:12 30 bp\n" + ("GSHSM" + "Q" * 177) + "\n"
            ">HLA:HLA02937 A*02:120 30 bp\n" + ("GSHSM" + "L" * 177) + "\n"
        )
        lookup = parse_imgt_fasta(text)
        self.assertEqual(lookup["HLA-A*02:12"][5], "Q")
        self.assertEqual(lookup["HLA-A*02:120"][5], "L")
        self.assertNotIn("HLA-A*02:12", [k for k in lookup if k.endswith("120")])

    def test_c67s(self):
        groove = "C" * 182
        mutated = apply_engineered("HLA-B*14:02(C67S)", groove)
        self.assertEqual(mutated[66], "S")
        self.assertEqual(mutated[65], "C")
        self.assertEqual(apply_engineered("HLA-B*14:02", groove)[66], "C")


if __name__ == "__main__":
    unittest.main()
