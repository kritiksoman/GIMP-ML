import sys

import torch

from _model_base import ModelBase, handle_alpha


class DeblurGANv2(ModelBase):
    def __init__(self):
        super().__init__()
        self.hub_repo = 'valgur/DeblurGANv2'

    def load_model(self):
        return torch.hub.load(self.hub_repo, 'predictor', 'fpn_inception', device=self.device)

    @handle_alpha
    @torch.no_grad()
    def predict(self, img):
        h, w, d = img.shape
        assert d == 3, "Input image must be RGB"
        return self.model(img, None)


model = DeblurGANv2()

if __name__ == '__main__':
    rpc_url = sys.argv[1]
    model.process_rpc(rpc_url)
