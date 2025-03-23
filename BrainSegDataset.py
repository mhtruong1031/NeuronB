import torch
from torch.utils.data import Dataset

class BrainSetDataset(Dataset):
    def __init__(self, inputs: list, outputs: list) -> None:
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        x = torch.tensor(self.inputs[index], dtype=torch.float32)
        y = torch.tensor(self.outputs[index], dtype=torch.float32)

        return x, y