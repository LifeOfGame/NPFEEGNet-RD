import unittest
import numpy as np
import torch
from scipy import signal
from model.factory import build_model
from training.engine import loss_value


class FactoryTests(unittest.TestCase):
    def test_all_baselines_backward_and_roundtrip(self):
        torch.manual_seed(2)
        x=torch.randn(2,1,20,1000)
        for name in ['eegnet','fbcnet','atcnet','conformer']:
            with self.subTest(model=name):
                settings=dict(model_name=name,num_channels=20,classes=2,n_times=1000,centering='demean')
                model=build_model(settings)
                logits,aux=model(x,return_aux=True)
                loss=loss_value(logits,aux,torch.tensor([0,1]));loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                model.eval();restored=build_model(settings).eval();restored.load_state_dict(model.state_dict())
                with torch.no_grad():torch.testing.assert_close(model(x),restored(x))

    def test_fbc_causal_fft_matches_direct_lfilter(self):
        settings=dict(model_name='fbcnet',num_channels=3,classes=2,n_times=256,centering='none')
        model=build_model(settings)
        rng=np.random.default_rng(8);x=rng.normal(size=(2,1,3,256))
        actual=model.core.filter(torch.tensor(x)).detach().numpy()
        for i,lo in enumerate(range(4,40,4)):
            wp=np.array([lo,lo+4])/125;ws=np.array([lo-2,lo+6])/125
            order,_=signal.cheb2ord(wp,ws,3,30)
            b,a=signal.cheby2(order,30,ws,btype='bandpass')
            expected=signal.lfilter(b,a,x[:,0],axis=-1)
            np.testing.assert_allclose(actual[:,i],expected,atol=1e-7,rtol=1e-6)

    def test_centering_modes(self):
        x=torch.randn(2,1,3,64);shift=torch.randn(2,1,3,1)*4
        model=build_model(dict(num_channels=3,classes=2,n_times=64,centering='full')).eval()
        with torch.no_grad():torch.testing.assert_close(model(x),model(x+shift),atol=1e-5,rtol=1e-5)
        with self.assertRaises(ValueError):build_model(dict(num_channels=3,classes=2,centering='bad'))


if __name__=='__main__':unittest.main()
