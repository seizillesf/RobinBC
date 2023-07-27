from icesheet_functions import *
from functools import partial
from multiprocessing import Pool
from itertools import product


# KEY PARAMETERS OF THE STUDY

K = 5 # 'dimension' of the parameters (nb of coefficients to estimate)


problem ="Laplace"

reg_type = 'matern' # {'matern' or 'squared_exp'}
alpha = 1 # parameter for Matern type regularisation 
r = 1 # parameter for the squared exponential family of priors

N = 100 # number of points at the surface
scale_noise = 0.01

size_msh_coarse = 30
size_msh_fine = 80


# Other notable parameters, fixed
domain_length = 5 # ratio of the rectangular domain, X/Y

gamma = 1e-7
reg = {'reg_type': 'matern', 'reg_parameter': 1}


#-----------------------------------DATA GENERATION----------------------------------------------------------

# Generate K coefficients following prior assumptions
theta_0 = np.random.normal(loc=0.0, scale=1, size=K)
theta_0 = np.array([-0.57495621,  0.68064092,  2.01914048,  0.11380695, -0.07830671])
print(f"Values of the true coefficients set to {theta_0}")



# Build the true function from coefficients

beta_0 = build_beta(theta_0)
interval = [(x,0) for x in np.linspace(0,1,100)]

# Plot
plt.figure(figsize=(15,5))
plt.title("True value of the basal drag - sampled on 100 pts")
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

plt.plot(covariates, observations, label='Noiseless observations')
plt.plot(covariates, noiseless_observations, label='Noisy observations')
plt.title("Surface observations")
plt.legend()
plt.show()



# Partial function with fixed arguments: only variables are N and regularisation type
def mcmc_for_parallel(N):
    return partial(run_mcmc_pcn, 
                            problem = problem, 
                            gamma=1e-7, 
                            n_iter =500000, 
                            size_msh=80, 
                            theta_0=theta_0,
                            observations=observations, 
                            covariates = covariates, 
                            scale_noise=scale_noise,
                            reg=reg)



def run_parallel():
    
    # list of ranges
    list_N = [100, 1000]
    #list_reg =[ {'reg_type': 'matern', 'reg_parameter': 1},
     #          {'reg_type': 'squared_exp', 'reg_parameter': 1}]
               
    # pool object with number of elements in the list
    with Pool(2) as p:
        print('coucou')
        p.map(mcmc_for_parallel, list_N)
    # map the function to the list and pass 
    # function and list_ranges as arguments
        
    
    

run_mcmc_pcn(problem = problem, N=100, gamma=1e-7, n_iter =500000, size_msh=80, theta_0=theta_0,observations=observations, covariates = covariates, scale_noise=scale_noise, reg=reg)