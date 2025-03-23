import torch, os

import nibabel as nib
import numpy as np

from Model import UNet

from inference import FINAL_MODEL_PATH
from preprocessing import normalize_data
from visualizer import render_from_angles

def run_pipeline(flair_path: str, t1Gd_path: str):
    # Load model
    model = UNet(2, 1)
    model.load_state_dict(torch.load(FINAL_MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    flair = nib.load(flair_path).get_fdata()
    t1Gd  = nib.load(t1Gd_path).get_fdata()

    flair = normalize_data(flair)
    t1Gd  = normalize_data(t1Gd)

    pred_stack = []

    for i in range(flair.shape[2]):
        flair_slice = flair[:, :, i]
        t1Gd_slice = t1Gd[:, :, i]

        input_tensor = np.stack([flair_slice, t1Gd_slice], axis = 0)
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0)  # [1, 2, 240, 240]

        with torch.no_grad():
            output = model(input_tensor)  # [1, 1, 240, 240]
            pred = (torch.sigmoid(output) > 0.5).float().squeeze().cpu().numpy()  # [240, 240]


        pred_stack.append(pred)
    
    pred_volume = np.stack(pred_stack, axis=-1)
    urls = render_from_angles(pred_volume, flair)

    return {
        "images": {
            "front":     "https://neuron-fb1rahtem-minhs-projects-c25c3719.vercel.app/front.png",
            "side":      "https://neuron-fb1rahtem-minhs-projects-c25c3719.vercel.app/side.png",
            "bottom":    "https://neuron-fb1rahtem-minhs-projects-c25c3719.vercel.app/bottom.png",
            "isometric": "https://neuron-fb1rahtem-minhs-projects-c25c3719.vercel.app/isometric.png",
        }
    }