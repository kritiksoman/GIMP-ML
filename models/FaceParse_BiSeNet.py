import sys

import numpy as np
import torch
import torch.hub
from PIL import Image
from torchvision import transforms

from _model_base import ModelBase, handle_alpha

colors = np.array([
    [0, 0, 0],
    [204, 0, 0],
    [0, 255, 255],
    [51, 255, 255],
    [51, 51, 255],
    [204, 0, 204],
    [204, 204, 0],
    [102, 51, 0],
    [255, 0, 0],
    [0, 204, 204],
    [76, 153, 0],
    [102, 204, 0],
    [255, 255, 0],
    [0, 0, 153],
    [255, 153, 51],
    [0, 51, 0],
    [0, 204, 0],
    [0, 0, 204],
    [255, 51, 153]
], dtype=np.uint8)


class FaceParse_BiSeNet(ModelBase):
    def __init__(self):
        super().__init__()
        self.hub_repo = 'valgur/face-parsing.PyTorch'

    def load_model(self):
        net = torch.hub.load(self.hub_repo, 'BiSeNet', pretrained=True, map_location=self.device)
        net.to(self.device)
        return net

    @handle_alpha
    @torch.no_grad()
    def predict(self, input_image):
        h, w, d = input_image.shape
        assert d == 3, "Input image must be RGB"

        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        input_image = Image.fromarray(input_image)
        img = input_image.resize((512, 512), Image.BICUBIC)
        img = to_tensor(img)
        img = torch.unsqueeze(img, 0)
        img = img.to(self.device)

        out = self.model(img)[0]

        result = out.squeeze(0).argmax(0).byte().cpu().numpy()
        result = Image.fromarray(result).resize(input_image.size)
        result.putpalette(colors.tobytes())
        result = np.array(result.convert('RGB'))
        return result


model = FaceParse_BiSeNet()

if __name__ == '__main__':
    rpc_url = sys.argv[1]
    model.process_rpc(rpc_url)
