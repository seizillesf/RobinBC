import pickle
from icesheet_functions import *
import matplotlib
import matplotlib.pyplot as plt



# Parameters we're interested in 
problem = 'Laplace'
reg_type = 'matern'
N = 100
gamma = 1e-7
scale_noise = 0.01

filename = f"experiment_PCN_{problem}_{reg_type}_{N}_{gamma}_{scale_noise}.pkl"



with open(filename, 'rb') as f:
    experiment = pickle.load(f)
    
    
# Retrieve values of parameters

size_msh_fine = experiment['size_mesh']
theta_0 = experiment['theta_0']
observations = experiment['observations']
covariates = experiment['covariates']

chain = experiment['coarse_chain']
likelihoods = experiment['coarse_likelihood']
accept_reject = experiment['coarse_acceptance']

n_iter = len(chain)



### ACCEPTANCE PROBABILITY


# Analysis of the acceptance probability:

print("Average acceptance probability: ", sum(accept_reject)/len(accept_reject))

# Plot moving average of the acceptance probability (should stabilise as iteration increase)

def moving_average(l, width=50):
    """
    Compute the moving average of list l
    
    Keyword arguments:
    l -- float list, list of which we want to compute the moving average
    width -- int, window size of the moving average
    
    Returns:
    list of moving averages
    """
    return np.convolve(l, np.ones(width), 'valid') / width


movingAverage = moving_average([int(e) for e in accept_reject])

plt.figure(figsize=(20,10))
plt.ylim(0,1)
plt.plot(movingAverage)
plt.title("Evolution of the accept ratio")
plt.xlabel('iterations')
plt.show()



### LIKELIHOODS

# Plot the evolution of the likelihood
plt.figure(figsize=(20,15))
plt.plot(likelihoods)
#print(likelihoods)

# Plot the true likelihood for reference
#true_likelihood = compute_loglikelihood(evaluate(), observations)
#plt.plot([true_likelihood]*n_iter, color = 'red')

plt.title("Evolution of the likelihood")




### CHAIN VALUES

# Plot the chain so we can visually assess if converges

fig = plt.figure(figsize=(15,15))

plt.subplot(3, 2, 1)
plt.title('theta_0')
plt.xlabel('chain iteration')
plt.plot([theta[0] for theta in chain])
plt.hlines(theta_0[0],0,n_iter,'r')
#plt.ylim(-10,10)

plt.subplot(3, 2, 2)
plt.title('theta_1')
plt.plot([theta[1] for theta in chain])
plt.xlabel('chain iteration')
plt.hlines(theta_0[1],0,n_iter,'r')
#plt.ylim(-10,10)

plt.subplot(3, 2, 3)
plt.title('theta_2')
plt.plot([theta[2] for theta in chain])
plt.xlabel('chain iteration')
plt.hlines(theta_0[2],0,n_iter,'r')
#plt.ylim(-10,10)

plt.subplot(3, 2, 4)
plt.title('theta_3')
plt.plot([theta[3] for theta in chain])
plt.xlabel('chain iteration')
plt.hlines(theta_0[3],0,n_iter,'r')
#plt.ylim(-10,10)

plt.subplot(3, 2, 5)
plt.title('theta_4')
plt.plot([theta[4] for theta in chain])
plt.xlabel('chain iteration')
plt.hlines(theta_0[4],0,n_iter,'r')
#plt.ylim(-10,10)

plt.show()



#  Plot histogram of chain, with mean + 95% confidence interval
import seaborn as sns
import scipy.stats as st

burnin = 100000
#burnin=0
#chain = chain[:30000]

# Compute stats
confidence_intervals=[]
means = []
for k in range(K):
    
    # Define sample data
    data = [theta[k] for theta in chain][burnin:]
    means.append(np.mean(data))
  
    # Create 95% confidence interval
    confidence_interval = st.t.interval(confidence=0.95, df=len(data)-1, loc=np.mean(data),scale=np.std(data))
    confidence_intervals.append(confidence_interval)


#---------------------------------Plot histograms----------------------------------------
fig = plt.figure(figsize=(20,15))

plt.subplot(3, 2, 1)
plt.title('theta_0')
sns.histplot([theta[0] for theta in chain][burnin:], kde = True, bins = 30)
plt.plot([theta_0[0], theta_0[0]],[0,0], 'ro', markersize=15, label = "Ground truth")
plt.plot([confidence_intervals[0][0], confidence_intervals[0][1]],[0,0], 'b^', markersize=15, label = "95% CI")
plt.plot([means[0], means[0]],[0,0], 'kP', markersize=15, label = "Chain mean")
plt.legend()

plt.subplot(3, 2, 2)
plt.title('theta_1')
sns.histplot([theta[1] for theta in chain][burnin:], kde = True, bins = 30)
plt.plot([theta_0[1], theta_0[1]],[0,0], 'ro', markersize=15, label = "Ground truth")
plt.plot([confidence_intervals[1][0], confidence_intervals[1][1]],[0,0], 'b^', markersize=15, label = "95% CI")
plt.plot([means[1], means[1]],[0,0], 'kP', markersize=15, label = "Chain mean")
plt.legend()


plt.subplot(3, 2, 3)
plt.title('theta_2')
sns.histplot([theta[2] for theta in chain][burnin:], kde = True, bins = 30)
plt.plot([theta_0[2], theta_0[2]],[0,0], 'ro', markersize=15, label = "Ground truth")
plt.plot([confidence_intervals[2][0], confidence_intervals[2][1]],[0,0], 'b^', markersize=15, label = "95% CI")
plt.plot([means[2], means[2]],[0,0], 'kP', markersize=15, label = "Chain mean")
plt.legend()


plt.subplot(3, 2, 4)
plt.title('theta_3')
sns.histplot([theta[3] for theta in chain][burnin:], kde = True, bins = 30)
plt.plot([theta_0[3], theta_0[3]],[0,0], 'ro', markersize=15, label = "Ground truth")
plt.plot([confidence_intervals[3][0], confidence_intervals[3][1]],[0,0], 'b^', markersize=15, label = "95% CI")
plt.plot([means[3], means[3]],[0,0], 'kP', markersize=15, label = "Chain mean")
plt.legend()

plt.subplot(3, 2, 5)
plt.title('theta_4')
sns.histplot([theta[4] for theta in chain][burnin:], kde = True, bins = 30)
plt.plot([theta_0[4], theta_0[4]],[0,0], 'ro', markersize=15, label = "Ground truth")
plt.plot([confidence_intervals[4][0], confidence_intervals[4][1]],[0,0], 'b^', markersize=15, label = "95% CI")
plt.plot([means[4], means[4]],[0,0], 'kP', markersize=15, label = "Chain mean")
plt.legend()

plt.show()


# Auxiliary function to determine whether the chain sample fits in the 95% confidence interval

def within_confidence_interval(sample, chain):
    """
    Determines whether the sample sits within the confidence interval - to exclude extreme betas in reconstructions
    
    Keyword arguments:
    sample -- K-dim list, values of coefficients, drawn from the chain
    chain -- the whole chain
    
    Returns:
    boolean - whether all K components of the chain sample are within the 95% confidence interval
    """
    
    
    # Run the loop over dimensions of the parameter
    for k in range(K):
        
        data=[theta[k] for theta in chain]
        #print(data)
    
        # Create 95% confidence interval
        confidence_interval = st.t.interval(confidence=0.95, df=len(data)-1, loc=np.mean(data),scale=np.std(data))
        
        #print(confidence_interval)
        within_interval = confidence_interval[0] <= sample[k] and confidence_interval[1] >= sample[k]
        
        if not within_interval:
            #print(k, confidence_interval[0], sample[k], confidence_interval[1] )
            return False
    
    return True





# Comparing true and simulated beta function

import random

#-------------Start plotting with the true function-----------------------

# Build the true function
beta_0 = build_beta(theta_0, reg_type='none')
interval = [(x,0) for x in np.linspace(0,1,100)]

plt.figure(figsize=(15,5))
plt.suptitle("Reconstruction of the basal drag - sampled on mesh coordinates")

plt.title("Beta(x)")
plt.plot([x[0] for x in interval], [beta_0(x) for x in interval], label = "Ground truth")



#-------------Add reconstruction from samples drawn from the chain---------------------------------

n_draws = 100
all_betas = [0 for x in interval] # table to keep track of betas and compute mean at the end
all_log_betas = [0 for x in interval] # table to keep track of log betas and compute mean at the end
drawn = 0

while drawn < n_draws:

    draw = random.choice(chain)
    #print(drawn, draw)
    
    
    if within_confidence_interval(draw, chain[burnin:]):

        beta = build_beta(draw)
        beta_values = [beta(x) for x in interval]
        
        plt.plot([x[0] for x in interval], beta_values, linewidth = 0.05, color='r')
        
     
        all_betas = [sum(x) for x in zip(all_betas, beta_values)] # update the list of all drawn betas
        
        drawn+=1 # update the count
        
        

# Add the empirical average to the plots

all_betas = [b/drawn for b in all_betas]
plt.plot([x[0] for x in interval], all_betas, color='g', label = "Mean from MCMC")
plt.legend()


plt.show()