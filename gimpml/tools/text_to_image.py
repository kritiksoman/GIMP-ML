import requests
import base64
import numpy as np
import cv2
import os
import shutil
import json
from PIL import Image


class TextToImage:
    def __init__(self, *args, **kwargs) -> None:
        self.gen_url = "https://api.openai.com/v1/images/generations"
        self.OPENAI_API_KEY = json.load(open(os.path.join(os.path.dirname(__file__), "..", "config.json")))['openai']['key']
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.OPENAI_API_KEY
        }    

    def gen_image(self, text="a photo of a happy corgi puppy sitting and facing forward, studio light, longshot",
                  model_version='standard',
                  output_size=None):
 
        if 0.9<=output_size[0]/output_size[1]<=1.1:
            s = [1024, 1024]
        elif output_size[0]>1.1*output_size[1]:
            s = [1024, 1792]
        else:
            s = [1792, 1024]

        data = {
            "prompt": text,
            "n": 1,
            "size": str(s[1]) + "x" + str(s[0]),
            "model": "dall-e-3",
            "quality": model_version
        }

        response = requests.post(self.gen_url, headers=self.headers, data=json.dumps(data))
        response_data = response.json()

        try:
            url = response_data['data'][0]['url']  # Replace with your image URL
            response = requests.get(url, stream=True)
            response.raise_for_status()
            self.image = np.array(Image.open(response.raw))
            # if output_size:
            #     self.image = cv2.resize(self.image, (output_size[1], output_size[0]))
        except Exception as e:
            print(e)
            return {"status": "failed"}
        return {"status": "success", "image": self.image}

    def get_gimp2_output(self):
        return base64.b64encode(self.image), self.image.shape
    
    def get_gimp3_output(self):
        return base64.b64encode(self.image.flatten().tobytes()), self.image.shape


if __name__ == "__main__":
    import sys
    models = ["Dalle3"]
    if len(sys.argv) >= 1:
        model_name = sys.argv[1]
    else:
        model_name = models[0]

    try:
        image_gen = TextToImage(model_name)
    except:
        pass
    filename = "generated_image.png"
    image_gen.gen_image()
    image_gen.save_image(filename)
