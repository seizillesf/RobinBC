
# In this scipt, we set the parameters of the experiment (problem, prior type, number of observations...) and run it. 
# For every experiment, a dictionary is created and stored in a .pkl file

# Imports
from icesheet_functions import *
from functools import partial
from multiprocessing import Pool
from itertools import product
import matplotlib
import matplotlib.pyplot as plt


# KEY PARAMETERS OF THE STUDY

K = 5 # 'dimension' of the parameters (nb of coefficients to estimate)

problem ="Laplace"
method = '' # either {'pCN', 'adaptive', 'LDA', 'combined'}. pCN for classical pCN,
            # 'adaptive' for pCN with adaptive step size, 'LDA' for the multilevel approach,
            # 'combined' for combining the multilevel Monte-Carlo with adative step size

reg_type = 'matern' # prior type, {'matern' or 'squared_exp'} 
reg_parameter = 1 # alpha for Matern type regularisation, r for the squared exponential
reg = {'reg_type': reg_type, 'reg_parameter': reg_parameter}

N = 100 # number of points at the surface
scale_noise = 0.1

size_msh_coarse = 30
size_msh_fine = 80


# Other notable parameters, fixed
domain_length = 5 # ratio of the rectangular domain, X/Y
gamma = 1e-7 # starting step size for pCN


#-----------------------------------DATA GENERATION----------------------------------------------------------

# Defining the ground truth

theta_0 = np.array([-0.6,  0.7,  2,  0.1, -0.08])
print(f"Values of the true coefficients set to {theta_0}")


# Build the true basal drag function from coefficients

beta_0 = build_beta(theta_0)
interval = [(x,0) for x in np.linspace(0,1,100)]

# Plot 
plt.figure(figsize=(15,5))
plt.title("True value of the basal drag")
plt.xlabel('x')
plt.ylabel("Beta(x)")
plt.plot([x[0] for x in interval], [beta_0(x) for x in interval])
plt.show()


# Draw the covariates (points at the surface where measurements are realised)

covariates = sorted(np.random.uniform(0,1,N)) # random N points where the solution is evaluated


### Generating synthetic data

# Retrieve the true values according to the model with true coefficients

noiseless_observations = forward_map(problem, theta_0, size_msh=size_msh_fine, covariates=covariates) # retrieve the true solution

added_noise = np.random.normal(loc=0.0, scale=scale_noise, size=np.shape(noiseless_observations))
observations = noiseless_observations + added_noise

plt.plot(covariates, observations, label='Noisy observations')
plt.plot(covariates, noiseless_observations, label='Noiseless observations')
plt.title("Surface observations")
plt.legend()
plt.show()


#--------------------Call to the function depending on the preferred method:------------------------

if method == 'pCN':
    run_mcmc_pcn(problem = problem, N=N, gamma=1e-7, n_iter =500000, size_msh=80, theta_0=theta_0,observations=observations, covariates = covariates, scale_noise=scale_noise, reg=reg)
    
elif method == 'adaptive':
    run_mcmc_adaptive(problem = problem, N=N, gamma=gamma, n_iter =500000, size_msh=80, theta_0=theta_0,observations=observations, covariates = covariates, scale_noise=scale_noise, reg=reg)
    
elif method == 'LDA':
    run_mcmc_lda(problem = problem, N=N, gamma=gamma, n_iter =200000, size_msh_fine=80, size_msh_coarse=30, theta_0=theta_0,observations=observations, covariates = covariates, scale_noise=scale_noise, reg=reg)
    
elif method == 'combined':
    run_mcmc_combined(problem = problem, N=N, gamma=gamma, n_iter =200000, size_msh_fine=80, size_msh_coarse=30, theta_0=theta_0,observations=observations, covariates = covariates, scale_noise=scale_noise, reg=reg)