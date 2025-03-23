import torch, os

import nibabel as nib
import numpy as np

from Model import UNet
from preprocessing import normalize_data, DATASET_PATH
from visualizer import render_from_angles, plot_surface_vedo

FINAL_MODEL_PATH = 'final_model_2.pth'

def main() -> None:
    # Load model
    model = UNet(2, 1)
    model.load_state_dict(torch.load(FINAL_MODEL_PATH, map_location=torch.device('cpu')))
    model.eval() # set to validation mode

    # Prep sample
    sample_id = 5396
    pred_volume, flair = predict_volume_from_sample(sample_id, model)

    plot_surface_vedo(pred_volume, flair)
    
# Returns a predicted tumor volume from a sample
def predict_volume_from_sample(sample_id: int, model) -> np.stack:
    folder = f'{DATASET_PATH}TCGA-CS-{str(sample_id)}/'
    for file in os.listdir(folder): # Grab flair and t1Gd files
            if 'flair' in file:
                flair_path = folder + file
            elif 't1Gd' in file:
                t1Gd_path = folder + file

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
    return pred_volume, flair

if __name__ == '__main__':
    main()