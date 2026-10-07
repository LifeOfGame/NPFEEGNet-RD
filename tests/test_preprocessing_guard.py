from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class PreprocessingGuardTests(unittest.TestCase):
    def test_target_mat_is_not_opened_without_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            result=subprocess.run([sys.executable,str(ROOT/'scripts/prepare_openbmi.py'),
                '--mat',str(Path(folder)/'sess02_subj01_EEG_MI.mat'),'--subject','1',
                '--session','2','--output',str(Path(folder)/'out')],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Session 2 requires completed source selection',result.stderr)
            self.assertFalse((Path(folder)/'out').exists())


if __name__=='__main__':unittest.main()
