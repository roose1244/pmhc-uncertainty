import unittest

import numpy as np
import pandas as pd

from src.uncertainty import conformal_quantile, fit_intervals, selective_mae


class ConformalTests(unittest.TestCase):
    def test_quantile_level(self):
        scores = np.arange(1, 101, dtype=float)
        q = conformal_quantile(scores, alpha=0.1)
        self.assertGreaterEqual(q, 91)
        self.assertLessEqual(q, 92)

    def test_intervals_use_calib_only(self):
        rng = np.random.default_rng(0)
        rows = []
        for fold, n in (("train", 20), ("val", 30), ("calib", 40), ("test", 25)):
            y = rng.normal(size=n)
            rows.append(
                pd.DataFrame(
                    {
                        "y_true": y,
                        "y_mean": y + rng.normal(scale=0.2, size=n),
                        "y_std": np.full(n, 0.2),
                        "fold": fold,
                    }
                )
            )
        scored, params = fit_intervals(pd.concat(rows, ignore_index=True))
        self.assertTrue(np.isfinite(params["q_norm"]))
        self.assertIn("lo", scored.columns)
        self.assertEqual(len(scored), 115)

    def test_selective_prefers_low_score(self):
        y = np.array([0.0, 0.0, 0.0, 0.0])
        yhat = np.array([0.0, 0.1, 1.0, 2.0])
        score = np.array([0.1, 0.2, 0.8, 0.9])
        out = selective_mae(y, yhat, score)
        self.assertLess(out["mae_50"], out["mae_100"])


if __name__ == "__main__":
    unittest.main()
