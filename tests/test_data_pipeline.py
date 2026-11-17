import unittest

import pandas as pd

from src.data_pipeline import build_features


class FeatureEngineeringTests(unittest.TestCase):
    def test_features_use_reference_maps_and_o_level_year_gap(self):
        applicants = pd.DataFrame(
            {
                "Mean Grade": [" B+ ", "D"],
                "O LEVEL YEAR": [2024, 2018],
                "County": [" Nairobi ", "Unknown County"],
                "Parent Status": ["Both Parents", "Orphan"],
                "High School": ["Example School", "Unlisted School"],
                "Apply_Year": [2025, 2025],
            }
        )

        features = build_features(
            applicants,
            {"nairobi": 1},
            {"example school": "NATIONAL"},
        )

        self.assertEqual(list(features.columns), [
            "Apply_Year",
            "mean_grade_score",
            "County Rate",
            "high_school_category_score",
            "o_level_apply_score",
            "parent_status_score",
        ])
        self.assertEqual(features.iloc[0].tolist(), [2025, 8, 1, 2, 5, 1])
        self.assertTrue(pd.isna(features.iloc[1]["County Rate"]))
        self.assertEqual(features.iloc[1]["high_school_category_score"], 4)
        self.assertEqual(features.iloc[1]["o_level_apply_score"], 1)
        self.assertEqual(features.iloc[1]["parent_status_score"], 4)


if __name__ == "__main__":
    unittest.main()
