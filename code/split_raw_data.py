import glob
import os
import shutil
dir_out = "/home/zm/WSL4MIS/data/ACDC_training/"
base_dir = "/home/zm/WSL4MIS/training/training/"
all_data = os.listdir(base_dir)
for file in all_data:
    print(file)
    if "patient" in file:
        data_dir = os.path.join(base_dir, file)
        for data in os.listdir(data_dir):
            if "_gt" in data:
                filepath = os.path.join(data_dir,data)
                shutil.copy(filepath, dir_out)
               