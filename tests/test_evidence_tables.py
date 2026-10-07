from fractions import Fraction
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_evidence_tables import combine, paired


def part(namespace,subject,correct):
    rows=[dict(subject=subject,seed=s,trials=2,correct=correct) for s in range(3)]
    arrays=dict(labels=np.zeros(6,dtype=int),probabilities=np.tile([.6,.4],(6,1)))
    return namespace,arrays,rows


class StudyTablesTests(unittest.TestCase):
    def test_distinct_cohorts_preserve_overlapping_ids(self):
        summary,means=combine([part('s',1,2),part('a',1,0)],2)
        self.assertEqual(set(means),{'s:1','a:1'})
        self.assertEqual(summary['accuracy_pct'],50)
        self.assertEqual(summary['runs'],6)

    def test_same_cohort_duplicate_partition_rejected(self):
        with self.assertRaises(ValueError):
            combine([part('s',1,2),part('s',1,2)],1)

    def test_incomplete_seeds_rejected(self):
        namespace,arrays,rows=part('s',1,2)
        with self.assertRaises(ValueError):
            combine([(namespace,arrays,rows[:2])],1)

    def test_exact_tie_and_paired_identity(self):
        result=paired({'s':Fraction(286,600)},{'s':Fraction(143,300)})
        self.assertEqual((result['wins'],result['ties'],result['losses']),(0,1,0))
        with self.assertRaises(ValueError):
            paired({'s':Fraction(1,2)},{'a':Fraction(1,2)})


if __name__=='__main__': unittest.main()
