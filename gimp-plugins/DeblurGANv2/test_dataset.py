import os
import unittest
from shutil import rmtree
from tempfile import mkdtemp

import cv2
import numpy as np
from torch.utils.data import DataLoader

from dataset import PairedDataset


def make_img():
    return (np.random.rand(100, 100, 3) * 255).astype('uint8')


class AugTest(unittest.TestCase):
    tmp_dir = mkdtemp()
    raw = os.path.join(tmp_dir, 'raw')
    gt = os.path.join(tmp_dir, 'gt')

    def setUp(self):
        for d in (self.raw, self.gt):
            os.makedirs(d)

        for i in range(5):
            for d in (self.raw, self.gt):
                img = make_img()
                cv2.imwrite(os.path.join(d, f'{i}.png'), img)

    def tearDown(self):
        rmtree(self.tmp_dir)

    def dataset_gen(self, equal=True):
        base_config = {'files_a': os.path.join(self.raw, '*.png'),
                       'files_b': os.path.join(self.raw if equal else self.gt, '*.png'),
                       'size': 32,
                       }
        for b in ([0, 1], [0, 0.9]):
            for scope in ('strong', 'weak'):
                for crop in ('random', 'center'):
                    for preload in (0, 1):
                        for preload_size in (0, 64):
                            config = base_config.copy()
                            config['bounds'] = b
                            config['scope'] = scope
                            config['crop'] = crop
                            config['preload'] = preload
                            config['preload_size'] = preload_size
                            config['verbose'] = False
                            dataset = PairedDataset.from_config(config)
                            yield dataset

    def test_equal_datasets(self):
        for dataset in self.dataset_gen(equal=True):
            dataloader = DataLoader(dataset=dataset,
                                    batch_size=2,
                                    shuffle=True,
                                    drop_last=True)
            dataloader = iter(dataloader)
            batch = next(dataloader)
            a, b = map(lambda x: x.numpy(), map(batch.get, ('a', 'b')))

            np.testing.assert_allclose(a, b)

    def test_datasets(self):
        for dataset in self.dataset_gen(equal=False):
            dataloader = DataLoader(dataset=dataset,
                                    batch_size=2,
                                    shuffle=True,
                                    drop_last=True)
            dataloader = iter(dataloader)
            batch = next(dataloader)
            a, b = map(lambda x: x.numpy(), map(batch.get, ('a', 'b')))

            assert not np.all(a == b), 'images should not be the same'
