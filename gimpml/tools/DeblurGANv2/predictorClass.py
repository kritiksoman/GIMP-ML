from models.networks import get_generator_new

# from aug import get_normalize
import torch
import numpy as np

config = {
    "project": "deblur_gan",
    "warmup_num": 3,
    "optimizer": {"lr": 0.0001, "name": "adam"},
    "val": {
        "preload": False,
        "bounds": [0.9, 1],
        "crop": "center",
        "files_b": "/datasets/my_dataset/**/*.jpg",
        "files_a": "/datasets/my_dataset/**/*.jpg",
        "scope": "geometric",
        "corrupt": [
            {
                "num_holes": 3,
                "max_w_size": 25,
                "max_h_size": 25,
                "name": "cutout",
                "prob": 0.5,
            },
            {"quality_lower": 70, "name": "jpeg", "quality_upper": 90},
            {"name": "motion_blur"},
            {"name": "median_blur"},
            {"name": "gamma"},
            {"name": "rgb_shift"},
            {"name": "hsv_shift"},
            {"name": "sharpen"},
        ],
        "preload_size": 0,
        "size": 256,
    },
    "val_batches_per_epoch": 100,
    "num_epochs": 200,
    "batch_size": 1,
    "experiment_desc": "fpn",
    "train_batches_per_epoch": 1000,
    "train": {
        "preload": False,
        "bounds": [0, 0.9],
        "crop": "random",
        "files_b": "/datasets/my_dataset/**/*.jpg",
        "files_a": "/datasets/my_dataset/**/*.jpg",
        "preload_size": 0,
        "corrupt": [
            {
                "num_holes": 3,
                "max_w_size": 25,
                "max_h_size": 25,
                "name": "cutout",
                "prob": 0.5,
            },
            {"quality_lower": 70, "name": "jpeg", "quality_upper": 90},
            {"name": "motion_blur"},
            {"name": "median_blur"},
            {"name": "gamma"},
            {"name": "rgb_shift"},
            {"name": "hsv_shift"},
            {"name": "sharpen"},
        ],
        "scope": "geometric",
        "size": 256,
    },
    "scheduler": {"min_lr": 1e-07, "name": "linear", "start_epoch": 50},
    "image_size": [256, 256],
    "phase": "train",
    "model": {
        "d_name": "double_gan",
        "disc_loss": "wgan-gp",
        "blocks": 9,
        "content_loss": "perceptual",
        "adv_lambda": 0.001,
        "dropout": True,
        "g_name": "fpn_inception",
        "d_layers": 3,
        "learn_residual": True,
        "norm_layer": "instance",
    },
}


class Predictor:
    def __init__(self, weights_path, model_name="", cf=False):
        # model = get_generator(model_name or config['model'])
        model = get_generator_new(weights_path[0:-11])
        model.load_state_dict(
            torch.load(weights_path, map_location=lambda storage, loc: storage)["model"]
        )
        if torch.cuda.is_available() and not cf:
            self.model = model.cuda()
        else:
            self.model = model
        self.model.train(True)
        # GAN inference should be in train mode to use actual stats in norm layers,
        # it's not a bug
        # self.normalize_fn = get_normalize()

    @staticmethod
    def _array_to_batch(x):
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        return torch.from_numpy(x)

    def _preprocess(self, x, mask):
        # x, _ = self.normalize_fn(x, x)
        x = ((x.astype(np.float32) / 255) - 0.5) / 0.5
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

    def __call__(self, img, mask, ignore_mask=True, cf=False):
        (img, mask), h, w = self._preprocess(img, mask)
        with torch.no_grad():
            if torch.cuda.is_available() and not cf:
                inputs = [img.cuda()]
            else:
                inputs = [img]
            if not ignore_mask:
                inputs += [mask]
            pred = self.model(*inputs)
        return self._postprocess(pred)[:h, :w, :]
