import os, torch
import nibabel as nib
import numpy as np

from sklearn.model_selection import train_test_split

# Image shape: (240, 240, 155)
# Voxel spacing: [1. 1. 1.]

DATASET_PATH        = 'resources/PKG/BraTS-TCGA-LGG/segmentation_data/'
PROCESSED_DATA_PATH = 'resources/processed_data/'

def main() -> None:
    samples = [folder for folder in os.listdir(DATASET_PATH)]
    samples.remove('TCGA_LGG_radiomicFeatures.csv')
    samples.remove('.DS_Store')

    inputs  = []
    outputs = []

    # Iterate over each sample and load flair and t1Gd files
    for s in samples:
        sample_path = DATASET_PATH + s + '/' # path of the actual sample folder
        for file in os.listdir(sample_path):
            if 'flair' in file:
                flair = sample_path+file
            elif 't1Gd' in file:
                t1Gd = sample_path+file
            elif 'ManuallyCorrected' in file:
                glistr = sample_path+file
            
        s_inputs, s_outputs = sample_to_tensors(flair, t1Gd, glistr)
        inputs  += s_inputs
        outputs += s_outputs

    # 80-20 randomized split
    train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(inputs, outputs, test_size=0.2, random_state=16)

    torch.save(train_inputs, PROCESSED_DATA_PATH+"training_inputs.pt")
    torch.save(test_inputs, PROCESSED_DATA_PATH+"testing_inputs.pt")
    torch.save(train_outputs, PROCESSED_DATA_PATH+"training_outputs.pt")
    torch.save(test_outputs, PROCESSED_DATA_PATH+"testing_outputs.pt")
            
# Converts an MRI 3D sample into a input/output lists of sliced tensors
def sample_to_tensors(flair_path: str, t1Gd_path: str, glistr_path: str) -> list:
    flair = nib.load(flair_path).get_fdata()
    t1Gd  = nib.load(t1Gd_path).get_fdata()

    flair = normalize_data(flair)
    t1Gd  = normalize_data(t1Gd)

    mask = nib.load(glistr_path).get_fdata()
    binary_mask = (mask > 0).astype(np.float32) # Generate a binary mask for every non-zero value (positive read)

    input_tensors   = []
    target_tensors  = []

    for i in range(flair.shape[2]):
        flair_slice = flair[:, :, i]
        t1Gd_slice = t1Gd[:, :, i]
        mask_slice = binary_mask[:, :, i]

        if np.max(mask_slice) == 0:
            continue

        input_tensor = np.stack([flair_slice, t1Gd_slice], axis = 0)

        input_tensors.append(input_tensor)
        target_tensors.append(mask_slice)
        
    return input_tensors, target_tensors # (input, output)

def normalize_data(file):
    return (file - np.mean(file)) / np.std(file)

if __name__ == '__main__':
    main()