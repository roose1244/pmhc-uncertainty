import unittest
from pathlib import Path

import pandas as pd

from src.data import MASTER_PATH, load_raw
from src.splits import HLA_TEST, HLA_VAL, assert_split, build_all


class SplitContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.master = load_raw()
        cls.splits = build_all(cls.master)

    def test_each_split_passes_asserts(self):
        for name, frame in self.splits.items():
            summary = assert_split(self.master, frame, name)
            self.assertIn("train", summary["counts"])
            self.assertIn("test", summary["counts"])

    def test_random_leaks_peptides_on_purpose(self):
        summary = assert_split(self.master, self.splits["random"], "random")
        self.assertGreater(summary["peptide_leakage"], 0.90)

    def test_peptide_and_cluster_do_not_leak_peptides(self):
        for name in ("peptide", "cluster"):
            summary = assert_split(self.master, self.splits[name], name)
            self.assertEqual(summary["peptide_leakage"], 0.0)

    def test_hla_holdouts(self):
        merged = self.master.merge(self.splits["hla"], on="id")
        self.assertEqual(set(merged.loc[merged["fold"] == "test", "hla"]), {HLA_TEST})
        self.assertEqual(set(merged.loc[merged["fold"] == "val", "hla"]), {HLA_VAL})


class WrittenFilesTests(unittest.TestCase):
    def test_on_disk_files_if_present(self):
        if not MASTER_PATH.exists():
            self.skipTest("master.parquet not built yet")
        master = pd.read_parquet(MASTER_PATH)
        for name in ("random", "peptide", "cluster", "hla"):
            path = Path("data/splits") / f"{name}.parquet"
            self.assertTrue(path.exists(), path)
            assert_split(master, pd.read_parquet(path), name)


if __name__ == "__main__":
    unittest.main()
