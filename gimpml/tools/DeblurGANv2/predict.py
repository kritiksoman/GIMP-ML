import os
from glob import glob

# from typing import Optional

import cv2
import numpy as np
import torch
import yaml
from fire import Fire
from tqdm import tqdm

from aug import get_normalize
from models.networks import get_generator


class Predictor:
    def __init__(self, weights_path, model_name=""):
        with open("config/config.yaml") as cfg:
            config = yaml.load(cfg)
        model = get_generator(model_name or config["model"])
        model.load_state_dict(
            torch.load(weights_path, map_location=lambda storage, loc: storage)["model"]
        )
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model
        self.model.train(True)
        # GAN inference should be in train mode to use actual stats in norm layers,
        # it's not a bug
        self.normalize_fn = get_normalize()

    @staticmethod
    def _array_to_batch(x):
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        return torch.from_numpy(x)

    def _preprocess(self, x, mask):
        x, _ = self.normalize_fn(x, x)
        if mask is None:
            mask = np.ones_like(x, dtype=np.float32)
        else:
            mask = np.round(mask.astype("float32") / 255)

        h, w, _ = x.shape
        block_size = 32
        min_height = (h // block_size + 1) * block_size
        min_width = (w // block_size + 1) * block_size

        pad_params = {
            "mode": "constant",
            "constant_values": 0,
            "pad_width": ((0, min_height - h), (0, min_width - w), (0, 0)),
        }
        x = np.pad(x, **pad_params)
        mask = np.pad(mask, **pad_params)

        return map(self._array_to_batch, (x, mask)), h, w

    @staticmethod
    def _postprocess(x):
        (x,) = x
        x = x.detach().cpu().float().numpy()
        x = (np.transpose(x, (1, 2, 0)) + 1) / 2.0 * 255.0
        return x.astype("uint8")

    def __call__(self, img, mask, ignore_mask=True):
        (img, mask), h, w = self._preprocess(img, mask)
        with torch.no_grad():
            if torch.cuda.is_available():
                inputs = [img.cuda()]
            else:
                inputs = [img]
            if not ignore_mask:
                inputs += [mask]
            pred = self.model(*inputs)
        return self._postprocess(pred)[:h, :w, :]


def sorted_glob(pattern):
    return sorted(glob(pattern))


def main(
    img_pattern,
    mask_pattern=None,
    weights_path="best_fpn.h5",
    out_dir="submit/",
    side_by_side=False,
):

    imgs = sorted_glob(img_pattern)
    masks = (
        sorted_glob(mask_pattern) if mask_pattern is not None else [None for _ in imgs]
    )
    pairs = zip(imgs, masks)
    names = sorted([os.path.basename(x) for x in glob(img_pattern)])
    predictor = Predictor(weights_path=weights_path)

    # os.makedirs(out_dir)
    for name, pair in tqdm(zip(names, pairs), total=len(names)):
        f_img, f_mask = pair
        img, mask = map(cv2.imread, (f_img, f_mask))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pred = predictor(img, mask)
        if side_by_side:
            pred = np.hstack((img, pred))
        pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dir, name), pred)


if __name__ == "__main__":
    Fire(main)
