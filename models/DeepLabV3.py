import sys

import numpy as np
import torch
import torch.hub
from PIL import Image
from torchvision import transforms

from _model_base import ModelBase, handle_alpha


class DeepLabV3(ModelBase):
    def __init__(self):
        super().__init__()
        self.hub_repo = 'pytorch/vision'

    def load_model(self):
        model = torch.hub.load(self.hub_repo, 'deeplabv3_resnet101', pretrained=True)
        model.eval()
        model.to(self.device)
        return model

    @handle_alpha
    @torch.no_grad()
    def predict(self, input_image):
        h, w, d = input_image.shape
        assert d == 3, "Input image must be RGB"

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        h, w, _ = input_image.shape
        input_tensor = preprocess(input_image).unsqueeze(0).to(self.device)

        output = self.model(input_tensor)['out'][0]
        output_predictions = output.argmax(0)

        # apply a color palette, selecting a color for each class
        palette = np.array([2**25 - 1, 2**15 - 1, 2**21 - 1])
        colors = np.arange(21)[:, None] * palette
        colors = (colors % 255).astype(np.uint8)
        result = Image.fromarray(output_predictions.byte().cpu().numpy()).resize((w, h))
        result.putpalette(colors.tobytes())
        return np.array(result.convert('RGB'))


model = DeepLabV3()

if __name__ == '__main__':
    rpc_url = sys.argv[1]
    model.process_rpc(rpc_url)
