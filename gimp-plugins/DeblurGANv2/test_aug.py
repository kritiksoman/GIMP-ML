import unittest

import numpy as np

from aug import get_transforms


class AugTest(unittest.TestCase):
    @staticmethod
    def make_images():
        img = (np.random.rand(100, 100, 3) * 255).astype('uint8')
        return img.copy(), img.copy()

    def test_aug(self):
        for scope in ('strong', 'weak'):
            for crop in ('random', 'center'):
                aug_pipeline = get_transforms(80, scope=scope, crop=crop)
                a, b = self.make_images()
                a, b = aug_pipeline(a, b)
                np.testing.assert_allclose(a, b)
