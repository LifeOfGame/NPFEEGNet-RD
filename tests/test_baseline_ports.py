import unittest
import torch
from baselines.openbmi_atcnet import OpenBMIATCNet
from baselines.openbmi_eegconformer import OpenBMIEEGConformer


class BaselinePortTests(unittest.TestCase):
    def test_openbmi_shapes_and_offset_invariance(self):
        torch.manual_seed(4)
        x = torch.randn(2, 1, 20, 1000)
        offset = torch.randn(2, 1, 20, 1) * 3
        for cls in [OpenBMIATCNet, OpenBMIEEGConformer]:
            with self.subTest(model=cls.__name__):
                model = cls(num_channels=20, classes=2, input_standardization='demean').eval()
                with torch.no_grad():
                    y = model(x)
                    shifted = model(x + offset)
                self.assertEqual(y.shape, (2, 2))
                self.assertTrue(torch.isfinite(y).all())
                torch.testing.assert_close(y, shifted, atol=1e-5, rtol=1e-5)
                if cls is OpenBMIATCNet:
                    self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 113338)


if __name__ == '__main__':
    unittest.main()
