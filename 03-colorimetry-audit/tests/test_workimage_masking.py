import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

from workimage_masking import (
    discover_pairs,
    masked_pixel_ita,
    neutral_fill,
    valid_skin_mask,
)


class WorkimageMaskingTests(unittest.TestCase):
    def test_discover_pairs_matches_identifiers_and_rejects_missing_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (2, 2)).save(root / "000001_lesion.png")
            Image.new("L", (2, 2), 255).save(root / "000001_mask.png")

            pairs = discover_pairs(root)

            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0].identifier, "000001")
            self.assertEqual(pairs[0].image_path.name, "000001_lesion.png")
            self.assertEqual(pairs[0].mask_path.name, "000001_mask.png")

            Image.new("RGB", (2, 2)).save(root / "000002_lesion.png")
            with self.assertRaisesRegex(ValueError, "000002_mask.png"):
                discover_pairs(root)

    def test_valid_skin_mask_excludes_only_masked_or_exact_black_pixels(self):
        rgb = np.array(
            [
                [[0, 0, 0], [1, 1, 1]],
                [[20, 10, 5], [200, 150, 120]],
            ],
            dtype=np.uint8,
        )
        supplied_mask = np.array([[255, 255], [0, 255]], dtype=np.uint8)

        keep = valid_skin_mask(rgb, supplied_mask)

        np.testing.assert_array_equal(
            keep,
            np.array([[False, True], [False, True]]),
        )

    def test_masked_pixel_ita_is_invariant_to_added_black_border(self):
        skin = np.full((10, 12, 3), [180, 120, 90], dtype=np.uint8)
        skin_mask = np.full((10, 12), 255, dtype=np.uint8)
        bordered = np.zeros((30, 32, 3), dtype=np.uint8)
        bordered_mask = np.zeros((30, 32), dtype=np.uint8)
        bordered[10:20, 10:22] = skin
        bordered_mask[10:20, 10:22] = 255

        plain = masked_pixel_ita(skin, skin_mask)
        with_border = masked_pixel_ita(bordered, bordered_mask)

        self.assertTrue(plain["valid"])
        self.assertTrue(with_border["valid"])
        self.assertAlmostEqual(plain["ita"], with_border["ita"], places=6)
        self.assertEqual(plain["valid_pixels"], with_border["valid_pixels"])

    def test_neutral_fill_preserves_skin_and_replaces_excluded_pixels(self):
        rgb = np.array(
            [
                [[0, 0, 0], [10, 20, 30]],
                [[30, 40, 50], [50, 60, 70]],
            ],
            dtype=np.uint8,
        )
        keep = np.array([[False, True], [True, False]])

        filled = neutral_fill(rgb, keep)

        np.testing.assert_array_equal(filled[keep], rgb[keep])
        np.testing.assert_array_equal(
            filled[~keep],
            np.array([[20, 30, 40], [20, 30, 40]], dtype=np.uint8),
        )

    def test_large_masked_image_converts_without_numeric_warnings(self):
        rgb = np.full((700, 730, 3), [180, 120, 90], dtype=np.uint8)
        supplied_mask = np.full((700, 730), 255, dtype=np.uint8)
        rgb[:100] = 0
        supplied_mask[:100] = 0

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = masked_pixel_ita(rgb, supplied_mask)

        self.assertTrue(result["valid"])


if __name__ == "__main__":
    unittest.main()
