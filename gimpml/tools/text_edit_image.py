import requests
import base64
import numpy as np
import cv2
import os
import shutil
import json
from PIL import Image
import openai
import matplotlib
import platform
import io
if platform.system() == 'Darwin':
    matplotlib.use('MacOSX')
from matplotlib import pyplot as plt


class TextEditImage:
    def __init__(self, *args, **kwargs) -> None:
        self.folder = os.path.join(os.path.dirname(__file__), "..", "__cache__")
        os.environ['OPENAI_API_KEY'] = json.load(open(os.path.join(os.path.dirname(__file__), "..", "service", "config.json")))['openai']['key']
    
    @staticmethod
    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


    def edit_image(self, image, text, mask=None, output_size=None):
        
        s = [1024, 1024]
        # print(image.shape, mask.shape)
        # image = np.copy(image)
        m = 255*(1-(mask[:, :, :] == [0, 0, 0, 255]).all(axis=2).astype(np.uint8))
        mask = np.copy(image)
        if mask.shape[2] == 4: # image already has alpha channel
            mask[:, :, 3] = m
        else:
            mask = np.dstack([mask, m])
        client = openai.OpenAI()
        # image_output = io.BytesIO()
        # image = Image.fromarray(image)
        # image.save(image_output, format='PNG')
        mask_output = io.BytesIO()        
        mask = Image.fromarray(mask)
        mask.save(mask_output, format='PNG')        

        response = client.images.edit(model="dall-e-2",
                                        image=mask_output,
                                        mask=mask_output,
                                        prompt=text,
                                        n=1,
                                        size="1024x1024"
                                    )     
        try:
            url = response.data[0].url
            response = requests.get(url, stream=True)
            response.raise_for_status()
            self.image = np.array(Image.open(response.raw))
            if output_size:
                self.image = cv2.resize(self.image, (output_size[1], output_size[0]))
        except Exception as e:
            print(e)
            return {"status": "failed"}
        return {"status": "success", "image": self.image}

    # def save_image(self, filename="generated_image.png"):
    #     self.image.save(filename)

    def get_gimp2_output(self):
        return base64.b64encode(self.image), self.image.shape
    
    def get_gimp3_output(self):
        return base64.b64encode(self.image.flatten().tobytes()), self.image.shape
    
    def save_image(self, path):
        cv2.imwrite(path, self.image[:, :, ::-1])
    

if __name__ == "__main__":
    try:
        image_gen = TextEditImage("dall-e-2")
    except:
        pass
    # image_gen.edit_image(r"/Users/kritiksoman/gimp-test copy/Untitled.png", 
    #                      r"/Users/kritiksoman/gimp-test copy/mask.png", 
    #                      "a photo of a cat with red glasses")
    image_gen.edit_image(r"/Users/kritiksoman/gimp-test copy/Untitled2.png", 
                        "new york in rainy night with bus in the front")
    image_gen.save_image(r"/Users/kritiksoman/gimp-test copy/op.png")

