import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sca3_compass.transcriptomics import (
    analyze_donor_profiles,
    benjamini_hochberg,
)


class TranscriptomicsTests(unittest.TestCase):
    def test_bh_adjustment_is_bounded_and_ordered(self) -> None:
        values = np.array([0.04, 0.001, 0.03, 0.2])
        adjusted = benjamini_hochberg(values)
        self.assertTrue(np.all((adjusted >= 0) & (adjusted <= 1)))
        order = np.argsort(values)
        self.assertTrue(np.all(np.diff(adjusted[order]) >= 0))

    def test_analysis_uses_donors_not_region_files(self) -> None:
        samples = []
        columns = []
        for genotype in ("SCA3", "CTRL"):
            for donor in range(4):
                regions = ("dorsal", "ventral") if donor == 0 else ("punch",)
                for region in regions:
                    accession = f"{genotype}-{donor}-{region}"
                    columns.append(accession)
                    samples.append(
                        {
                            "accession": accession,
                            "genotype": genotype,
                            "sex": "F" if donor % 2 == 0 else "M",
                            "time": str(40 + donor),
                            "pmi": f"{10 + donor} hr",
                            "tissue type": region,
                            "donor_id": f"{genotype}-{donor}",
                        }
                    )
        matrix = pd.DataFrame(
            {
                column: [8.0, 2.0, 1.0] if column.startswith("SCA3") else [1.0, 2.0, 8.0]
                for column in columns
            },
            index=pd.Index(["ENST1.1", "ENST2.1", "ENST3.1"], name="target_id"),
        )
        results, pca, metrics = analyze_donor_profiles(matrix, samples)
        self.assertEqual(metrics["sca3_donors"], 4)
        self.assertEqual(metrics["control_donors"], 4)
        self.assertEqual(len(pca), 8)
        self.assertGreater(
            results.set_index("transcript_version").loc["ENST1.1", "log2_fold_change"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
