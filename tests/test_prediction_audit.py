import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('audit_predictions', ROOT/'scripts/audit_predictions.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class PredictionAuditTests(unittest.TestCase):
    def test_bin_boundary_and_pooling(self):
        # 0.5 belongs to (0.4, 0.5], 0.6 to (0.5, 0.6].
        p = np.array([[.5, .5], [.4, .6], [.4, .6]])
        nll, ece = audit.calibration(np.array([0, 1, 0]), p)
        self.assertAlmostEqual(nll, -np.log([.5, .6, .4]).mean())
        self.assertAlmostEqual(ece, .5/3 + .1*2/3)

    def test_reject_unnormalized(self):
        with self.assertRaises(ValueError):
            audit.calibration(np.array([0]), np.array([[.7, .7]]))

    def test_original_primary_outputs(self):
        prefix = ROOT/'evidence/predictions'
        rd, _ = audit.summarize(prefix/'paper-20260917__openbmi_final_npf_raw_demean_3seed.npz')
        no, _ = audit.summarize(prefix/'paper-20260917__openbmi_final_npf_nodemean_3seed.npz')
        self.assertEqual(rd['runs'], 162)
        self.assertEqual(rd['trials'], 32400)
        self.assertAlmostEqual(rd['subject_macro_accuracy_pct'], 70.6358024691358)
        self.assertAlmostEqual(no['subject_macro_accuracy_pct'], 66.4506172839506)
        self.assertAlmostEqual(rd['pooled_nll'], .63633714176785)
        self.assertAlmostEqual(no['pooled_nll'], .83594920056053)


if __name__ == '__main__':
    unittest.main()
