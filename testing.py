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
from sbi.analysis import pairplot
from sbi.analysis import plot_summary
from sbi.utils import BoxUniform
from sbi.diagnostics import run_sbc
from sbi.analysis.plot import sbc_rank_plot

with open('cca_compressed_data.pkl', 'rb') as f:
    load = pickle.load(f)
    print('The compressed parameter and data pairs are as below', load)
    param_samples = load['params']
    cls_samples = load['compressed_cls']
    
def cca(param_samples, cls_samples, parameters, data):
    '''
    Compute the sampled parameter auto covariance, simulated data vector auto covariance and the parameter-data vector cross covariance
    Methodology as per Park, M., Gatti, M., & Jain, B. (2025). Dimensionality reduction techniques for statistical inference in cosmology.
        - output has the shape of n_samples x n_params
    '''
    cov = np.cov(param_samples.T, cls_samples.T)
    param_count = param_samples.shape[1]
    cp = cov[:param_count,:param_count]
    ct = cov[param_count:,param_count:]
    ctp = cov[param_count:,:param_count]
    # print('Cp',cp.shape,'Ct',ct.shape,'Ctp',ctp.shape)

    cl = ctp@np.linalg.inv(cp).T@ctp.T
    # print('Cl',cl.shape,np.linalg.matrix_rank(cl))


    # print((np.linalg.inv(ct)@cl).shape)
    # print(np.linalg.matrix_rank(np.linalg.inv(ct)@cl))

    # Using scipy's eigh function for generalized eigenvalue problem, requires symmetric matrices and positive definite ct-cl
    e_vals, e_vecs = sp.linalg.eigh(ct, ct - cl)
    print('e_vals',e_vals.shape,'e_vecs',e_vecs.shape)
    canon_corr = e_vals[::-1][:param_count]
    canon_projs = e_vecs[:,::-1][:,:param_count]
    
    # Using numpy's eig function for generalized eigenvalue problem, less efficient but works for non-symmetric matrices)
    # e_vals, e_vecs = np.linalg.eig(np.linalg.inv(ct)@cl)
    # print('e_vals',e_vals.shape,'canon_projs',canon_projs.shape)

    plt.plot(np.arange(0,param_count),e_vals[:(-1*param_count-1):-1])
    plt.title('Canonical correlations')
    plt.xlabel('Component number')
    plt.gca().xaxis.get_major_locator().set_params(integer=True)
    plt.ylabel('Correlation')

    plt.show()
    
    return {'params': parameters, 'compressed_cls': data@canon_projs}

#%%
h_range = (0.6, 0.8)
Oc_range = (0.2, 0.4)
Ob_range = (0.03, 0.05)
lower_bound = torch.as_tensor([0.6, 0.2, 0.03])
upper_bound = torch.as_tensor([0.8, 0.4, 0.05])
prior = BoxUniform(low=lower_bound, high=upper_bound)

with open('coverage_data.pkl', 'rb') as f:  
    load = pickle.load(f)
    prior_samples = torch.tensor(load['params'], dtype=torch.float32, device='mps')
    prior_predictives = torch.tensor(load['cls'], dtype=torch.float32, device='mps')

with open('NLE_NSF_uni1000.pkl', 'rb') as f:
    load = pickle.load(f)
    NLE_uni1000 = load
    posterior = NLE_uni1000.build_posterior(prior=prior) 

#We need to compress the prior_predictives using the same compression scheme as before
with open('sbi_demo_data.pkl', 'rb') as f:
    load = pickle.load(f)
    param_samples = load['params']
    cls_samples = load['cls']

cca_result = cca(param_samples, cls_samples, prior_samples.cpu().numpy(), prior_predictives.cpu().numpy())
print('compressed dataset has shape of',cca_result['compressed_cls'].shape)

# run SBC: for each inference we draw 1000 posterior samples.
num_posterior_samples = 200
ranks, dap_samples = run_sbc(
    prior_samples,
    torch.tensor(cca_result['compressed_cls'], device='mps', dtype=torch.float32),
    posterior,
    reduce_fns=lambda theta, x: -posterior.log_prob(theta, x),
    num_posterior_samples=num_posterior_samples,
    use_batched_sampling=False,  # `True` can give speed-ups, but can cause memory issues.
)
fig, ax = sbc_rank_plot(
    ranks,
    num_posterior_samples,
    plot_type="cdf",
    num_bins=20,
    figsize=(5, 3),
)