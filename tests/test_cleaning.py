import unittest

import pandas as pd

from sentiment import clean_data


class CleaningTests(unittest.TestCase):
    def test_removes_empty_neutral_and_invalid_labels(self):
        frame = pd.DataFrame({
            "InputText": ["  behi  ", "khayeb", "neutre", "", None, "unknown"],
            "SentimentLabel": ["1", -1, 0, 1, -1, "invalid"],
        })
        result = clean_data(frame)
        self.assertEqual(result["InputText"].tolist(), ["behi", "khayeb"])
        self.assertEqual(result["SentimentLabel"].tolist(), [1, -1])

    def test_removes_conflicting_texts_before_deduplication(self):
        frame = pd.DataFrame({
            "InputText": ["same", " same ", "same", "good", "good"],
            "SentimentLabel": [1, -1, 1, 1, 1],
        })
        result = clean_data(frame)
        self.assertEqual(result["InputText"].tolist(), ["good"])

    def test_does_not_modify_original_frame(self):
        frame = pd.DataFrame({"InputText": [" a "], "SentimentLabel": ["1"]})
        original = frame.copy(deep=True)
        clean_data(frame)
        pd.testing.assert_frame_equal(frame, original)

    def test_missing_columns_have_clear_error(self):
        with self.assertRaisesRegex(ValueError, "SentimentLabel"):
            clean_data(pd.DataFrame({"InputText": ["behi"]}))


if __name__ == "__main__":
    unittest.main()
