import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
from PIL import Image
import numpy as np
import os

class HeartMRIDataset_DICOM(Dataset):
    def __init__(self, images, labels):
        self.image_filenames = images
        self.label_filenames = labels
        print ("images:", len((self.image_filenames)), "#labels:", len((self.label_filenames)))

        
    def __getitem__(self, index):
        label_path = self.label_filenames[index]

        label = np.load(label_path) 
        label = label[None, ...]
        label = torch.tensor(label)

        label_filename = label_path.split("/")[-1]
        image_file_path = self.image_filenames[index]
        if (os.path.basename(label_path) != os.path.basename(image_file_path)): 
            print ("LABEL AND IMAGE DON'T MATCH")

        image = np.load(image_file_path)
        image = image[None, ...]
        image = torch.tensor(image).float()

        patient_id, filename_only = label_filename.split(".npy")[0].split("_") 
        return {"image": image, "label": label, "patient_id" : patient_id, "file_id": filename_only} 

    def __len__(self):
        return len(self.label_filenames)

