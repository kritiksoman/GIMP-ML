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


class TextOutpaintImage:
    def __init__(self, *args, **kwargs) -> None:
        self.folder = os.path.join(os.path.dirname(__file__), "..", "__cache__")
        os.environ['OPENAI_API_KEY'] = json.load(open(os.path.join(os.path.dirname(__file__), "..", "service", "config.json")))['openai']['key']
    
    @staticmethod
    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


    def edit_image(self, image, text, side=None, output_size=None):
        print(side)
        gen_image = np.zeros((image.shape[0], image.shape[1], 4))
        
        s = [1024, 1024]
        

        if side == 'Right':
            gen_image[:, :256, :3] = image[:, 768:, :3]
            gen_image[:, :256, 3] = 255
            image_new = Image.new('RGBA', (1024+768, 1024), (0, 0, 0, 0))
            image = Image.fromarray(image).convert('RGBA')
            image_new.paste(image, (0, 0), image)
        elif side == 'Left':
            gen_image[:, 768:, :3] = image[:, :256, :3]
            gen_image[:, 768:, 3] = 255
            image_new = Image.new('RGBA', (1024+768, 1024), (0, 0, 0, 0))
            image = Image.fromarray(image).convert('RGBA')
            image_new.paste(image, (768, 0), image)
        elif side == 'Top':
            gen_image[768:, :, :3] = image[:256, :, :3]
            gen_image[768:, :, 3] = 255
            image_new = Image.new('RGBA', (1024, 1024+768), (0, 0, 0, 0))
            image = Image.fromarray(image).convert('RGBA')
            image_new.paste(image, (0, 768), image)
        elif side == 'Bottom':
            gen_image[:256, :, :3] = image[768:, :, :3]
            gen_image[:256, :, 3] = 255
            image_new = Image.new('RGBA', (1024, 1024+768), (0, 0, 0, 0))     
            image = Image.fromarray(image).convert('RGBA') 
            image_new.paste(image, (0, 0), image)  
        
        client = openai.OpenAI()
        image_output = io.BytesIO()
        image = Image.fromarray(gen_image.astype(np.uint8))
        image.save(image_output, format='PNG')
        response = client.images.edit(model="dall-e-2",
                                        image=image_output,
                                        mask=image_output,
                                        prompt=text,
                                        n=1,
                                        size="1024x1024"
                                    )     
        try:
            url = response.data[0].url
            response = requests.get(url, stream=True)
            response.raise_for_status()
            extended_image = Image.open(response.raw)
            extended_image.putalpha(Image.new('L', extended_image.size, 255))
            # extended_image = Image.open("tmp.png")
            if side == 'Right':
                image_new.paste(extended_image, (768, 0), extended_image)
            elif side == 'Left':       
                image_new.paste(extended_image, (0, 0), extended_image)     
            elif side == 'Top':
                image_new.paste(extended_image, (0, 0), extended_image) 
            elif side == 'Bottom':
                image_new.paste(extended_image, (0, 768), extended_image) 
            # image_new.save("tmp.png")
            # image_new = Image.open("tmp.png")
            self.image = np.array(image_new).astype(np.uint8)[:, :, :3]
            if output_size:
                self.image = cv2.resize(self.image, (output_size[1], output_size[0]))
        except Exception as e:
            print(e)
            return {"status": "failed"}
        return {"status": "success", "image": self.image}

    # def save_image(self, filename="generated_image.png"):
    #     self.image.save(filename)

    def get_gimp2_output(self):
        self.image = np.ascontiguousarray(self.image)
        return base64.b64encode(self.image), self.image.shape
    
    def get_gimp3_output(self):
        return base64.b64encode(self.image.flatten().tobytes()), self.image.shape
    
    def save_image(self, path):
        cv2.imwrite(path, self.image[:, :, ::-1])
    

if __name__ == "__main__":
    try:
        image_gen = TextOutpaintImage("dall-e-2")
    except:
        pass
    # image_gen.edit_image(r"/Users/kritiksoman/gimp-test copy/Untitled.png", 
    #                      r"/Users/kritiksoman/gimp-test copy/mask.png", 
    #                      "a photo of a cat with red glasses")
    image_gen.edit_image(r"/Users/kritiksoman/gimp-test copy/Untitled2.png", 
                        "new york in rainy night with bus in the front")
    image_gen.save_image(r"/Users/kritiksoman/gimp-test copy/op.png")

