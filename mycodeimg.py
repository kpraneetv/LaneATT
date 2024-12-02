import torch
import yaml
import json
from lib.config import Config
from lib.models import LaneATT
import cv2  
import numpy as np
import os  
import time 


def yaml_tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))

yaml.add_constructor('tag:yaml.org,2002:python/tuple', yaml_tuple_constructor)

def load_model(config_path, model_path):
    cfg = Config(config_path)
    backbone = cfg['model']['parameters'].get('backbone', 'resnet34')
    
    
    model = LaneATT(backbone=backbone, pretrained_backbone=True, S=cfg['model']['parameters']['S'],
                    img_w=cfg['model']['parameters']['img_w'], img_h=cfg['model']['parameters']['img_h'],
                    anchors_freq_path=cfg['model']['parameters'].get('anchors_freq_path', None),
                    topk_anchors=cfg['model']['parameters'].get('topk_anchors', None))
    
   
    checkpoint = torch.load(model_path, map_location="cpu")
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()  
    return model

def preprocess_image(image_path):
   
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 360))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
    return image_tensor

def run_inference(model, image_tensor):
    start_time = time.time()
    with torch.no_grad():
        output = model(image_tensor)  
    run_time = time.time() - start_time
    return output, run_time

def save_output_to_file(output, image_path, run_time, save_path="model_output.json"):
    
    lanes = [lane.cpu().numpy().tolist() for lane in output[0][0]]  

    result = {
        "raw_file": image_path,
        "lanes": lanes,
        "run_time": run_time
    }

   
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Output saved to {save_path}")

if __name__ == "__main__":
    config_path = "cfgs/laneatt_tusimple_resnet34.yml"
    model_path = os.path.join("experiments", "laneatt_r34_tusimple", "models", "model_0100.pt")
    image_path = "img_resize.png"
    model = load_model(config_path, model_path)
    image_tensor = preprocess_image(image_path)
    output, run_time = run_inference(model, image_tensor)
    print("Model output:", output)
    save_output_to_file(output, image_path, run_time, "model_output.json")
