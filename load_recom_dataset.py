import numpy as np
import scipy.io


data_name = 'user.npy'
out_name = data_name[:-4]
data_path = '../dataset_Large-Scale/new_dataset/recommendation/video_games/'

data = np.load(data_path+data_name)
scipy.io.savemat(data_path+out_name, dict(data=data))

print(data)
