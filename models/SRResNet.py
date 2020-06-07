import sys

import torch

from _model_base import ModelBase, handle_alpha


class SRResNet(ModelBase):
    def __init__(self):
        super().__init__()
        self.hub_repo = 'valgur/pytorch-SRResNet'

    def load_model(self):
        model = torch.hub.load(self.hub_repo, 'SRResNet', pretrained=True, map_location=self.device)
        model.to(self.device)
        return model

    @handle_alpha
    @torch.no_grad()
    def predict(self, input_image):
        h, w, d = input_image.shape
        assert d == 3, "Input image must be RGB"

        im_input = torch.from_numpy(input_image).permute(2, 0, 1).to(self.device)
        im_input = im_input.float().div(255).unsqueeze(0)

        HR_4x = self.model(im_input).squeeze().permute(1, 2, 0)
        return HR_4x.clamp(0, 1).mul(255).byte().cpu().numpy()


model = SRResNet()

if __name__ == '__main__':
    rpc_url = sys.argv[1]
    model.process_rpc(rpc_url)
