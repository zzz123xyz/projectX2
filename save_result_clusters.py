import numpy as np
import scipy.io


data_name = 'result_clusters.mat'
out_name = data_name[:-4]

# '../dataset_Large-Scale/new_dataset/recommendation/musical_instrument/'
# '../dataset_Large-Scale/new_dataset/recommendation/Office_Products/'
data_path = '../dataset_Large-Scale/new_dataset/recommendation/Office_Products/'

result_clusters = scipy.io.loadmat(data_path+data_name)
len(result_clusters)

np.save(data_path+out_name, result_clusters)

print(data_name)
