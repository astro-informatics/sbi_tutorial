#%%
# simulation script
from glass_sim import *

# general 
import scipy as sp
import pickle
from tqdm.notebook import tqdm

# density estimation 
from sbi.inference import NPE, NLE
import torch

#%%
with open('moped_compressed_data.pkl', 'rb') as f:
    load = pickle.load(f)
    print('The compressed parameter and data pairs are as below', load)
    param_samples = load['params']
    cls_samples = load['compressed_cls']
    
    print('The dimension of the parameter datset is', param_samples.shape)
    print('The dimension of the compressed data dataset is', cls_samples.shape)

#%%
inference  = NPE(density_estimator="nsf")

# Convert parameter and cls data to tensors for training
param_samples = torch.tensor(param_samples, dtype=torch.float32, requires_grad=True)
cls_samples = torch.tensor(cls_samples, dtype=torch.float32, requires_grad=True)

# Check whether the trainig data is of correct shape and network has been instantiated
print(param_samples.shape, cls_samples.shape)
print(inference)

#%%
posterior_model = inference.append_simulations(param_samples, cls_samples)
posterior_model.train()
