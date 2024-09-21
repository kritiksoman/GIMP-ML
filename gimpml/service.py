import  os
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools.text_to_image import TextToImage
from tools.text_edit_image import TextEditImage
from tools.text_extend_image import TextExtendImage
from tools.text_outpaint_image import TextOutpaintImage
from fastapi import FastAPI, Request, APIRouter
# import gradio as gr
# import torch
import uvicorn
import numpy as np
import base64
from matplotlib import pyplot as plt
import psutil
import os, platform, subprocess, re

# CUSTOM_PATH = "/gradio"

app = FastAPI()

class GimpMlService:
    def __init__(self) -> None:
        self.model = None
        self.model_name = None
        self.router = APIRouter()
        self.router.add_api_route("/status", self.status, methods=["GET"])
        self.router.add_api_route("/download_load_model", self.download_load_model, methods=["POST"])
        self.router.add_api_route("/run_inference", self.run_inference, methods=["POST"])

    @staticmethod
    def get_processor_name():
        # Function to get the CPU name
        try:
            if platform.system() == "Windows":
                return platform.processor()
            elif platform.system() == "Darwin":
                os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
                command ="sysctl -n machdep.cpu.brand_string"
                return subprocess.check_output(command.split(" ")).strip()
            elif platform.system() == "Linux":
                command = "cat /proc/cpuinfo"
                all_info = subprocess.check_output(command, shell=True).decode().strip()
                for line in all_info.split("\n"):
                    if "model name" in line:
                        return re.sub( ".*model name.*:", "", line,1).strip()
        except:
            return ""

    def status(self):
        # Function to get info about the system running the service
        cuda_total, cuda_used, cuda_free = 0, 0, 0
        ram_total, ram_used, ram_free = psutil.virtual_memory().total/(1024.**3), psutil.virtual_memory().used/(1024.**3), psutil.virtual_memory().free/(1024.**3)
        # cuda  = torch.cuda.is_available()
        cuda = False # TODO: change
        # if cuda:
        #     cuda_stats = torch.cuda.mem_get_info()
        #     cuda_total = cuda_stats[1]/(1024.**3)
        #     cuda_free = cuda_stats[0]/(1024.**3)
        #     cuda_used = cuda_total-cuda_free  # free inside reserved
        return {"service": "running",
                "cuda_available": cuda,
                "cuda_total": round(cuda_total, 2),
                "cuda_used": round(cuda_used, 2),
                "cuda_free": round(cuda_free, 2),
                "ram_total": round(ram_total, 2),
                "ram_used": round(ram_used, 2),
                "ram_free": round(ram_free, 2),
                "cpu": self.get_processor_name(),
                "os": platform.system()}

    async def download_load_model(self, request: Request):
        # Function to Download and load the model if not already loaded
        input_data = await request.json()
        # print(input_data)
        output_json = {}
        model_name = input_data["model"]
        if self.model_name == model_name:
            output_json["status"] = "Loaded."
        else:
            try:                
                output_json["status"] = "Loaded."
                if input_data["pipeline"] == "text_to_image":
                    self.model = TextToImage(input_data["model"])
                elif input_data["pipeline"] == "text_edit_image":
                    self.model = TextEditImage(input_data["model"])
                elif input_data["pipeline"] == "text_extend_image":
                    self.model = TextExtendImage(input_data["model"])
                elif input_data["pipeline"] == "text_outpaint_image":
                    self.model = TextOutpaintImage(input_data["model"])
                self.model_name = model_name
            except Exception as e:
                output_json["status"] = "Error!"
        return output_json
    
    @staticmethod
    def get_numpy_img(input_data, k="image"):
        img = input_data[k]
        img = base64.b64decode(img)
        im_shape = input_data[k + "_shape"]
        img_np = np.frombuffer(img, dtype=np.uint8)
        img_np = np.reshape(img_np, im_shape)
        return img_np
        

    async def run_inference(self, request: Request):
        # Function to run the inference from the model and get the prediction
        input_data = await request.json()
        # print(input_data)
        output = dict()
        if "pipeline" in input_data:
            # Generate output based on input, model and output
            if input_data["pipeline"] == "text_to_image":
                self.model.gen_image(input_data["text"], 
                                     model_version = input_data["model"], 
                                     output_size=input_data["image_shape"])
            elif input_data["pipeline"] == "text_edit_image":
                self.model.edit_image(self.get_numpy_img(input_data, "image"), 
                                     input_data["text"], 
                                     mask = self.get_numpy_img(input_data, "mask"), 
                                     output_size=input_data["image_shape"])
            elif input_data["pipeline"] == "text_extend_image":
                self.model.edit_image(self.get_numpy_img(input_data, "image"), 
                                     input_data["text"], 
                                     side = input_data["ext_side"], 
                                     )        
            elif input_data["pipeline"] == "text_outpaint_image":
                self.model.edit_image(self.get_numpy_img(input_data, "image"), 
                                     input_data["text"]
                                     )           
            if input_data["source"] == "gimp2":
                im_b64, im_shape = self.model.get_gimp2_output()
            else:
                im_b64, im_shape = self.model.get_gimp3_output()
            output["image"] = im_b64
            output["image_shape"] = im_shape
            output["text"] = input_data["text"]
        else:
            # return the same image for debugging
            img = input_data["image"]
            img = base64.b64decode(img)
            im_shape = input_data["image_shape"]
            img_np = np.frombuffer(img, dtype=np.uint8)
            img_np = np.reshape(img_np, im_shape)
            # img_np[20:20, 20:20, :] = 0
            # return
            ret_im = base64.b64encode(img_np.flatten().tobytes())
            output = {"image": ret_im}
        # torch.cuda.empty_cache()
        return output

# io = gr.Interface(lambda x: "Hello, " + x + "!", "textbox", "textbox")
# app = gr.mount_gradio_app(app, io, path=CUSTOM_PATH)

ml_service = GimpMlService()
app.include_router(ml_service.router)

if __name__ == "__main__":
    uvicorn.run("service:app", host="127.0.0.1", port=8000, log_level="info")

# uvicorn service:app

# python -m python_coreml_stable_diffusion.pipeline --prompt "a photo of an astronaut riding a horse on mars" -i /Users/kritiksoman/gimp-test/models/coreml-stable-diffusion-v1-4_original_packages/ -o output --compute-unit CPU_AND_GPU --seed 93


"""
GET:
http://127.0.0.1:8000/status


POST:
http://127.0.0.1:8000
{
    "pipeline": "StableDiffusionPipeline",
    "model": "CompVis/stable-diffusion-v1-4",
    "image": ""
}

"""

