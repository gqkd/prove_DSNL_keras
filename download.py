#%%
import os

# os.system("cd /download_dataset/data")
# os.system("chmod +x download_physionet.sh")
# os.system("./download_physionet.sh")
# os.system("python prepare_physionet_original.py --data_dir data --output_dir data/eeg_fpz_cz --select_ch 'EEG Fpz-Cz'")
# os.system("cd ../..")

pwd = os.getcwd()
log_dir="/logs/base_fit/"
path = pwd + log_dir
if not os.path.exists(path):
    os.makedirs(path)

from data_prep import data_loader
X_train, X_valid, y_train, y_valid = data_loader()



# %%
