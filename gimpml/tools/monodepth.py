# import torch
from transformers import DPTForDepthEstimation, DPTFeatureExtractor
import base64
import numpy as np
import cv2
import os
import shutil
from PIL import Image


class ImageToDepth:
    def __init__(self, model_id="Intel/dpt-hybrid-midas", device="cuda", model_path=os.path.join(os.path.expanduser("~"), "gimpml")) -> None:
        self.model_id = model_id
        self.model_path = os.path.join(model_path, model_id.replace("/","-"))
        # self.model_id = r"/home/kritik/pycharm_projects/models/stable-diffusion-2-1"
        self._load_save_model()
        self.device = device
        # self.pipe = StableDiffusionPipeline.from_pretrained(
        #     self.model_id, torch_dtype=torch.float16, cache_dir=model_path)
        # self.pipe = self.pipe.to(self.device)
        self.model.to(self.device)
        # self.feature_extractor.to(self.device)

    def _load_save_model(self):
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)
        
        try:
            self.model = DPTForDepthEstimation.from_pretrained(self.model_path, low_cpu_mem_usage=True, local_files_only=True)
            self.feature_extractor = DPTFeatureExtractor.from_pretrained(self.model_path, local_files_only=True)

            # self.pipe = StableDiffusionPipeline.from_pretrained(
            #             self.model_path, torch_dtype=torch.float16, local_files_only=True)
        except:
            try:
                self.model = DPTForDepthEstimation.from_pretrained(self.model_id, low_cpu_mem_usage=True, local_files_only=False)
                self.feature_extractor = DPTFeatureExtractor.from_pretrained(self.model_id, local_files_only=False)
                # self.pipe = StableDiffusionPipeline.from_pretrained(
                #             self.model_id, torch_dtype=torch.float16, local_files_only=False)
                self.model.save_pretrained(self.model_path)
                self.feature_extractor.save_pretrained(self.model_path)
                tmp_model_path = self.feature_extractor.model.model_name_or_path
                shutil.rmtree(tmp_model_path)
                # self.pipe = StableDiffusionPipeline.from_pretrained(
                #         self.model_path, torch_dtype=torch.float16, local_files_only=True)
                self.model = DPTForDepthEstimation.from_pretrained(self.model_path, low_cpu_mem_usage=True, local_files_only=True)
                self.feature_extractor = DPTFeatureExtractor.from_pretrained(self.model_path, local_files_only=True)
            except:
                raise FileNotFoundError("Failed to load model!")    
    



    def gen_image(self, input_image, output_size=None):
        image = Image.fromarray(input_image)
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        # visualize the prediction
        output = prediction.squeeze().cpu().numpy()
        self.image = (output * 255 / np.max(output)).astype("uint8")
        self.image = np.dstack([self.image, self.image, self.image])

        if output_size:
            self.image = cv2.resize(self.image, (output_size[1], output_size[0]))

    # def save_image(self, filename="generated_image.png"):
    #     self.image.save(filename)

    def get_gimp2_output(self):
        # im_np = np.array(self.image, dtype=np.uint8)
        return base64.b64encode(self.image), self.image.shape
    
    def get_gimp3_output(self):
        # im_np = np.array(self.image, dtype=np.uint8)
        return base64.b64encode(self.image.flatten().tobytes()), self.image.shape
    
    # def get_ui8encoded_string(self):
    #     im_np = np.array(self.image)
    #     return np.uint8(im_np).tobytes(), im_np.shape


# if __name__ == "__main__":
#     filename = "generated_image.png"
#     image_gen = TextToImage("CompVis/stable-diffusion-v1-4")
#     image_gen.gen_image()
#     # image_gen.save_image(filename)
# #     image_gen = TextToImage("stabilityai/stable-diffusion-2-1")
# #     image_gen.gen_image()
# #     image_gen.save_image(filename.replace(".png", "_tmp.png"))
     
#     # # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
#     # pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
#     # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
#     # pipe = pipe.to("cuda")

#     # prompt = "a photo of an astronaut riding a horse on mars"
#     # image = pipe(prompt).images[0]
        
#     # image.save("astronaut_rides_horse.png")
