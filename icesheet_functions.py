#!/usr/bin/env python
# coding: utf-8



# ## Preliminary

# Dimension of parameters space
K=5 #

# Parameters for the physical model

domain_length=5 #ratio X/Y
rho = 1 # constant ice density
gx, gy = 5, 5 # gravity component


#-----------------------------------CONSTRUCTION OF THE BASAL DRAG FUNCTION----------------------------------------------------- 

import numpy as np
import matplotlib
import matplotlib.pyplot as plt



def build_beta(coeffs):
    """
    Auxiliary function to build a function beta based on the theta coefficients in the arguments.
    We exponentiate because beta must be a positive function.

    Keyword arguments:
    coeffs -- float list of size K, the coefficients weigthing the truncated Fourier expansion functions

    Returns:
    beta -- one-dimensional function representing the basal drag factor
    """ 
    functions = [1]+[np.cos,np.sin]*(K//2) # 
    wavenumbers = [0]+[1+k//2 for k in range(K-1)] # wavenumbers inside the sinus function[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
    beta = lambda x: np.exp((coeffs[0] + sum([coeffs[k]*functions[k](wavenumbers[k]*2*np.pi*x[0]) for k in range(1,K)])))
    
    return beta






# ## ----------------------------------------PDE - Forward map----------------------------------------------------

# We define here the elements necessary to the building of the forward map, which maps the coefficients $\theta$ (array of size K) to the velocity, evaluated at N points drawn uniformly on the surface. We parameterise the forward map by the mesh size.


# NECESSARY IMPORTS

import dolfinx
import time
import pyvista

import ufl
from dolfinx import cpp as _cpp
from dolfinx import fem
from dolfinx import plot
from dolfinx.fem import (Constant, Function, FunctionSpace, dirichletbc,
                         extract_function_spaces, form, Expression,
                         locate_dofs_geometrical, locate_dofs_topological)
from dolfinx.io import XDMFFile
from dolfinx.mesh import (CellType, GhostMode, create_rectangle, locate_entities,
                          locate_entities_boundary, meshtags)

from dolfinx import geometry

from ufl import div, dx, grad, inner, Measure

from mpi4py import MPI
from petsc4py import PETSc



# AUXILIARY FUNCTIONS FOR FORWARD MAP

# Create mesh and facet tags for relevant boundaries

def create_mesh_and_tags(size_msh,domain_length=5):
    """
    Creates the mesh and facet tags for the PDE problem.
    
    Keyword arguments:
    size_msh -- int, size of mesh considered
    domain_length -- float, ratio of the rectangular considered (x in [0,1], y in [0, 1/domain_length])
    
    Returns:
    msh -- mesh object
    facet_tag -- facets based on specified boundary conditions
    """
    
    size_msh_x = size_msh*domain_length
    size_msh_y = size_msh

    msh = create_rectangle(MPI.COMM_WORLD,
                           [np.array([0, 0]), np.array([1, 1/domain_length])],
                           [size_msh_x, size_msh_y],
                           CellType.triangle, GhostMode.none)


    #Create the subdomains and space for Robin boundary condition in this problem
    tol = 1E-14 # tolerance (we cannot use strict equalities)
    boundaries = [(1, lambda x: abs(x[1])<= tol),   # Robin BC, at the bottom
                  (2, lambda x: abs(x[1]-1/domain_length)<= tol)]   # Neumann BC, at  the surface


    # Loop through all the boundary conditions and create MeshTags identifying the facets for each boundary condition
    facet_indices, facet_markers = [], []
    fdim = msh.topology.dim - 1

    for (marker, locator) in boundaries:
        facets = locate_entities(msh, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = meshtags(msh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])
    
    return msh, facet_tag


# Define weak forms for the Laplace and Stokes problems

def define_weak_form_laplace(theta, msh, facet_tag, V):
    """
    Creates the bilinear and linear forms for the weak form associated to the Robin Laplace problem
    
    Keyword arguments:
    theta -- array of size K, coefficients in the expansion
    msh -- mesh on which we solve the problem
    facet_tag -- facet tags, marking the boundaries
    V -- function space
    
    Returns
    a -- bilinear form in the weak formulation
    L -- linear form in the weak formulation
    bcs -- Dirichlet boundary conditions for the problem
    """
    
    ds = Measure("ds", domain=msh, subdomain_data=facet_tag)
    fdim = msh.topology.dim - 1

    
    # Define the class of boundary conditions 

    class BoundaryCondition():
        def __init__(self, type, marker, values):
            self._type = type
            if type == "Dirichlet":
                u_D = Function(V)
                u_D.interpolate(values)
                facets = facet_tag.find(marker)
                dofs = locate_dofs_topological(V, fdim, facets)
                self._bc = dirichletbc(u_D, dofs)

            elif type == "Neumann":
                self._bc = inner(values, v) * ds(marker)

            elif type == "Robin":
              #self._bc = values[0] * inner(u-values[1], v)* ds(marker)
              # slight modification: returns 2 integrals, one for the bilinear form a and one for the linear form L
                self._bc = values[0] * inner(u,v)* ds(marker), values[0] * inner(values[1], v)* ds(marker)
            else:
                raise TypeError("Unknown boundary condition: {0:s}".format(type))

        @property
        def bc(self):
            return self._bc

        @property
        def type(self):
            return self._type


    version = "sum"

    # Define variational problem: Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Define the source terms (based on tunable parameters at the top)
    f = Constant(msh, PETSc.ScalarType(0))

    tau_fct = lambda x: 10*(np.sin(12*np.pi*x[0])+1)
    tau = Function(V)
    tau.interpolate(tau_fct)

    # Define the bilinear form
    bilinear = inner(grad(u), grad(v)) * dx

    # Define the base linear form
    L = inner(f, v) * dx
    
    #-----------------SET THE BOUNDARY CONDITIONS FOR THE PROBLEM----------------------------------------------

    # Set the values for Neumann BC
    values_boundary_neumann = tau 

    # Set the values for Robin BC
    beta = build_beta(theta)
    
    r = Function(V)
    r.interpolate(beta)
    s = Constant(msh, PETSc.ScalarType(0))
    values_boundary_robin = (r,s)

    # Gather the Boundary conditions
    boundary_conditions = [BoundaryCondition("Robin", 1, values_boundary_robin),
                        BoundaryCondition("Neumann", 2, values_boundary_neumann)]

    bcs = []
    for condition in boundary_conditions:
        if condition.type == "Dirichlet":
            bcs.append(condition.bc)

        elif condition.type == "Neumann":
            linear_term = condition.bc
            L+= linear_term

        elif condition.type == "Robin":

            bilinear_term, linear_term = condition.bc

            if version == "sum":
                a = bilinear + bilinear_term

            else:
                a[0].append(bilinear_term) # add the modification to bilinear form


            L+= linear_term   # add the modification to linear form

        else: 
            print("Unhandled condition type")
    return a, L, bcs
    


def define_weak_form_stokes(theta, msh, facet_tag, W, V, Q):
    """
    Creates the bilinear and linear forms for the weak form associated to the Robin Laplace problem
    
    Keyword arguments:
    theta -- array of size K, coefficients in the expansion
    msh -- mesh on which we solve the problem
    facet_tag -- facet tags, marking the boundaries
    W, V, Q -- mixed function space / function space for velocity / function space for pressure
    
    Returns
    a -- bilinear form in the weak formulation
    L -- linear form in the weak formulation
    bcs -- Dirichlet boundary conditions for the problem
    """
    
    ds = Measure("ds", domain=msh, subdomain_data=facet_tag)
    fdim = msh.topology.dim - 1

    
    # Define the class of boundary conditions 

    class BoundaryCondition():
        def __init__(self, type, marker, values):
            self._type = type
            if type == "Dirichlet":
                u_D = Function(V)
                u_D.interpolate(values)
                facets = facet_tag.find(marker)
                dofs = locate_dofs_topological(V, fdim, facets)
                self._bc = dirichletbc(u_D, dofs)

            elif type == "Neumann":
                self._bc = inner(values, v) * ds(marker)

            elif type == "Robin":
              #self._bc = values[0] * inner(u-values[1], v)* ds(marker)
              # slight modification: returns 2 integrals, one for the bilinear form a and one for the linear form L
                self._bc = values[0] * inner(u,v)* ds(marker), values[0] * inner(values[1], v)* ds(marker)
            else:
                raise TypeError("Unknown boundary condition: {0:s}".format(type))

        @property
        def bc(self):
            return self._bc

        @property
        def type(self):
            return self._type


    version = "sum"





    # We now define the bilinear and linear forms corresponding to the weak
    # mixed formulation of the Stokes equations in a blocked structure:

    # Define variational problem: Trial and test functions
    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)

    # Define the source terms (based on tunable parameters at the top)
    f = Constant(msh, (PETSc.ScalarType(rho*gx), PETSc.ScalarType(rho*gy)))


    
    # Define the bilinear form
    bilinear = inner(grad(u), grad(v)) * dx - inner(p, div(v)) * dx + inner(div(u), q) * dx

    # Define the linear form
    L = inner(f, v) * dx + inner(Constant(msh, PETSc.ScalarType(0)), q) * dx



    #-----------------SET THE BOUNDARY CONDITIONS FOR THE PROBLEM----------------------------------------------

    # Set the values for Neumann BC at the surface
    tau_fct = lambda x: (10*(np.sin(12*np.pi*x[0])+1), 0*x[1])
    tau = Function(V)
    tau.interpolate(tau_fct)
    values_boundary_neumann = tau

    # Set the values for Robin BC at the bottom
    beta = build_beta(theta)
    r = Function(Q)
    r.interpolate(beta)
    s = Constant(msh, (PETSc.ScalarType(0), PETSc.ScalarType(0)))
    values_boundary_robin = (r,s)


    # Gather the Boundary conditions
    boundary_conditions = [BoundaryCondition("Robin", 1, values_boundary_robin),
                        BoundaryCondition("Neumann", 2, values_boundary_neumann)]

    bcs = []
    for condition in boundary_conditions:
        if condition.type == "Dirichlet":
            bcs.append(condition.bc)

        elif condition.type == "Neumann":
            linear_term = condition.bc
            L+= linear_term

        elif condition.type == "Robin":

            bilinear_term, linear_term = condition.bc

            if version == "sum":
                a = bilinear + bilinear_term

            else:
                a[0].append(bilinear_term) # add the modification to bilinear form


            L+= linear_term   # add the modification to linear form

        else: 
            print("Unhandled condition type")
    return a, L, bcs
    


# Functions to solve for the Laplace or the Stokes problem on the domain

def solve_laplace(theta, msh, facet_tag):
    """
    Maps the theta coefficients to the solution of the Laplace PDE problem

    Keyword arguments:
    theta -- array of size K, coefficients in the expansion of the basal drag function beta
    
    Returns:
    uh -- solution of the PDE problem equations 
    """
    
    #------------------------------------PREPARE THE SOLVE------------------------------------------------
        
    # Define the finite elements function space
    P1 = ufl.FiniteElement("Lagrange", msh.ufl_cell(), 1)
    V = FunctionSpace(msh, P1)

    
    #-----------------GET THE WEAK FORM----------------------------------------------
    a,L, bcs = define_weak_form_laplace(theta, msh,facet_tag, V)
     
    
    #-----------------------------ASSEMBLE AND SOLVE-----------------------------------------------

    # Assemble LHS matrix and RHS vector
    a,L = form(a),form(L)

    A = fem.petsc.assemble_matrix(a, bcs=bcs)
    A.assemble()
    b = fem.petsc.assemble_vector(L)

    fem.petsc.apply_lifting(b, [a], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Set Dirichlet boundary condition values in the RHS
    fem.petsc.set_bc(b, bcs)

    # Create and configure solver
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("superlu_dist")

    # Compute the solution
    uh = Function(V)
    ksp.solve(b, uh.vector)
    
    return uh


def solve_stokes(theta, msh, facet_tag):
    """
    Maps the theta coefficients to the solution of the Stokes PDE problem

    Keyword arguments:
    theta -- array of size K, coefficients in the expansion of the basal drag function beta
    
    Returns:
    uh -- solution of the PDE problem equations 
    """
    
    #------------------------------------PREPARE THE SOLVE------------------------------------------------
        
    # We define the finite elements function space (Taylor Woods method)
    P2 = ufl.VectorElement("Lagrange", msh.ufl_cell(), 2)
    P1 = ufl.FiniteElement("Lagrange", msh.ufl_cell(), 1)
    mixed = ufl.MixedElement([P2, P1])

    V, Q = FunctionSpace(msh, P2), FunctionSpace(msh, P1)
    W = FunctionSpace(msh, mixed) # Defined Mixed Function space - needed for solving divergence at same time
    

    
    #-----------------GET THE WEAK FORM----------------------------------------------
    a,L, bcs = define_weak_form_stokes(theta, msh,facet_tag, V)
     
    
    #-----------------------------ASSEMBLE AND SOLVE-----------------------------------------------

    # Assemble LHS matrix and RHS vector
    a,L = form(a),form(L)

    A = fem.petsc.assemble_matrix(a, bcs=bcs)
    A.assemble()
    b = fem.petsc.assemble_vector(L)

    fem.petsc.apply_lifting(b, [a], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # Set Dirichlet boundary condition values in the RHS
    fem.petsc.set_bc(b, bcs)

    # Create and configure solver
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("superlu_dist")

    # Compute the solution
    U = Function(W)
    
    start_time = time.time()
    ksp.solve(b, U.vector)
    solve_time = time.time()-start_time

    # Split the mixed solution and collapse
    uh = U.sub(0).collapse()
    ph = U.sub(1).collapse()
    
    return uh



# Function to evaluate the solution at specified coordinates

def evaluate(solution, covariates, msh, domain_length=5):
    """
    Evaluates the solution of the PDE at specified covariate points
    
    Keyword arguments:
    solution -- solution of the PDE problem equations, solved by FEM
    covariates -- list of coordinates on which the evaluation is performed. Does not have to match mesh points
    msh -- mesh
    domain_length -- float
    
    Returns:
    uv -- list, evaluations of the solution at covariates points
    """
    
    bb_tree = geometry.BoundingBoxTree(msh, msh.topology.dim)
    points = [[c, 1/domain_length, 0] for c in covariates] # from surface covariates to points

    # Find cells which bounding-box collide with the the point
    cell_candidates = dolfinx.geometry.compute_collisions(bb_tree, points)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(msh, cell_candidates, points)    

    #cell= colliding_cell.links(0)
    uv=solution.eval(points, colliding_cells.array)
    
    return uv


# FORWARD MAP - Putting together auxiliary functions

def forward_map(problem, theta, size_msh, covariates):
    
    msh, facet_tag = create_mesh_and_tags(size_msh)
    
    if problem=='Laplace':
        uh = solve_laplace(theta, msh, facet_tag)
        
    elif problem=='Stokes':
        uh = solve_stokes(theta, msh, facet_tag)
        
    else:
        raise Exception("Unknown problem")
        
    u_on_covariates = evaluate(uh, covariates, msh)
    
    return u_on_covariates



# ------------------------------MARKOV CHAIN MONTE CARLO----------------------------------------------------------

import pickle


# Auxiliary functions

def compute_loglikelihood(model_values, observations, scale_noise):
    """
    Keyword arguments:
    observations -- array, list of observations at the surface
    model_values -- array, outputs of the model (forward map) for the velocity at the surface

    Returns:
    loglikelihood -- int, the loglikelihood defined as in Richard Nickl's notes
    """
    
    loglikelihood = 0
    
    for i in range(len(model_values)):
        observation = observations[i]
        model_value = model_values[i]
        loglikelihood+= (np.linalg.norm(observation-model_value))**2
    
    loglikelihood = -1/2 * loglikelihood  / (scale_noise**2)

    return loglikelihood



def compute_proposal_pcn(current_value, gamma, reg):
    """
    Compute the proposal value for the coefficients of beta function for pCN scheme

    Keyword arguments:
    current_value -- array, current value of the Markov chain
    gamma -- float, step size
    reg_type -- 'matern' or 'square_exp', regularisation coefficient as defined in the paper

    Returns:
    proposal -- array, proposal for the next state of the Markov chain
    """
    #print("Current value: ", current_value)
    reg_type = reg['reg_type']
    reg_parameter = reg['reg_parameter']
    
    wavenumbers = [0]+[1+k//2 for k in range(K-1)] # wavenumbers inside the sinus function[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]

    if reg_type == 'matern':
        reg_coeffs = np.array([(1+wavenumbers[k]**2)**(-reg_parameter/2) for k in range(K)])
    
    elif reg_type== 'squared_exp':
        reg_coeffs = np.array([np.exp(-reg_parameter*(wavenumbers[k]**2)/2) for k in range(K)])
        
    step = np.multiply(reg_coeffs, np.random.normal(loc=0, scale=1, size=K))
    
    proposal = np.sqrt(1-2*gamma)*current_value + np.sqrt(2*gamma)*step
    #print("Proposal: ", proposal)

    return proposal



def metropolis_step(current_loglikelihood, proposal_loglikelihood, accept_reject):
    """
    Practice the Metropolis-Hastings step for the pCN scheme 
    
    Keyword arguments:
    current_loglikelihood -- int, current loglikelihood
    proposal_loglikelihood -- int, loglikelihood of the proposal

    Returns:
    boolean -- True if we accept the proposal, False otherwise
    """
    acceptance_probability = min(0, proposal_loglikelihood - current_loglikelihood)
    
    #print(f"Proposal likelihood: {proposal_loglikelihood}, Current likelihood: {current_loglikelihood}")
    #print("Accept proba:", acceptance_probability)
    # We draw a U([0,1]) r.v and return the proposal with probability acceptance_probability

    draw = np.random.uniform()
    #print(np.exp(acceptance_probability))

    return np.log(draw) < acceptance_probability







def run_mcmc_lda(problem, N, gamma, n_iter, reg, size_msh_fine, size_msh_coarse,theta_0,observations, covariates, scale_noise):
    
    # Create 'experiment' dictionary to keep track of runs and parameters
    experiment = {}
    experiment['problem'] = problem
    experiment['N'] = N
    experiment['gamma'] = gamma
    experiment['reg_type'] = reg['reg_type']
    experiment['size_mesh_fine'] = size_msh_fine
    experiment['size_mesh_coarse'] = size_msh_coarse
    experiment['theta_0'] = theta_0
    experiment['observations'] = observations
    experiment['covariates'] = covariates
    experiment['scale_noise'] = scale_noise
    
    # Create the filename in which to load the dictionary
    filename = f"experiment_LDA_{problem}_{reg['reg_type']}_{N}_{gamma}_{scale_noise}"
    
    
    counter = 0

    # INITIALISATION
    theta = theta_0 + np.random.normal(loc=0.0, scale=0.1, size=K) #initialise close to true value

    # Compute the model and likelihood of this proposal on coarse mesh
    model_values_coarse = forward_map(theta, size_msh = size_msh_coarse)
    current_loglikelihood_coarse = compute_loglikelihood(model_values_coarse, observations)
    
    # Compute the model and likelihood of this proposal on fine mesh
    model_values_fine = forward_map(theta, size_msh = size_msh_fine)
    current_loglikelihood_fine = compute_loglikelihood(model_values_fine, observations)

    
    # Useful tables to store values
    chain_coarse = np.array([np.array(theta)])
    chain_fine = np.array([np.array(theta)])
    
    
    likelihood_coarse = np.array([current_loglikelihood_coarse])
    likelihood_fine = np.array([current_loglikelihood_fine])
    
    accept_reject_coarse = np.array([True])
    accept_reject_fine = np.array([True])


    #--------------------ITERATE--------------------------------------------------
    while counter < n_iter: # effective sample size of 20000, we stop when finer chain has achieved n_iter accepted
    
        #-----------------------RUN ON COARSE CHAIN FIRST----------------------------------------

        # Get the proposal for new value
        proposal = compute_proposal_pcn(theta, gamma, reg)

        # Compute the model and likelihood of this proposal
        model_values = forward_map(problem, proposal, size_msh = size_msh_coarse, covariates=covariates)
        proposal_loglikelihood = compute_loglikelihood(model_values, observations,scale_noise)

        # Metropolis step
        accept = metropolis_step(current_loglikelihood_coarse, proposal_loglikelihood, accept_reject_coarse)
        
        # Update the coarse tables
        if accept:
            accept_reject_coarse = np.append(accept_reject_coarse,[True])
            theta = proposal.copy()
            current_loglikelihood_coarse = proposal_loglikelihood

        else:
            accept_reject_coarse = np.append(accept_reject_coarse,[False])

        chain_coarse = np.append(chain_coarse, np.array([theta]),axis=0)
        likelihood_coarse = np.append(likelihood_coarse, [current_loglikelihood_coarse])
    
        
        # Save runs in dictionary
        experiment['coarse_chain'] = np.array(chain_coarse)
        experiment['coarse_likelihood'] = np.array(likelihood_coarse)
        experiment['coarse_acceptance'] = np.array(accept_reject_coarse)
        with open(f'{filename}.pkl', 'wb') as fp:
            pickle.dump(experiment, fp) 

        print("Nb of coarse runs:", len(likelihood_coarse))
        
        #-----------------RUN ON FINER CHAIN IF ACCEPTED ON COARSE---------------------------
        
        if accept:
            
            # Use the proposal to compute likelihood on finer mesh
            model_values = forward_map(problem, proposal, size_msh = size_msh_fine, covariates=covariates)
            proposal_loglikelihood = compute_loglikelihood(model_values, observations, scale_noise)
            
            # Metropolis step
            accept = metropolis_step(current_loglikelihood_fine, proposal_loglikelihood, accept_reject_fine)

            if accept:
                counter+=1
                print(counter, np.sum(accept_reject_coarse))
                current_loglikelihood_fine = proposal_loglikelihood
                
                accept_reject_fine = np.append(accept_reject_fine,[True])
                chain_fine = np.append(chain_fine, np.array([proposal]),axis=0)
                #likelihood_fine = np.append(likelihood_fine, [current_loglikelihood_fine])            

            else:
                accept_reject_fine = np.append(accept_reject_fine,[False])
                chain_fine = np.append(chain_fine, np.array([chain_fine[-1,:]]),axis=0)
                   
                    
            likelihood_fine = np.append(likelihood_fine, [current_loglikelihood_fine])         
            
            # Save runs in dictionary
            experiment['fine_chain'] = np.array(chain_fine)
            experiment['fine_likelihood'] = np.array(likelihood_fine)
            experiment['fine_acceptance'] = np.array(accept_reject_fine)
            with open(f'{filename}.pkl', 'wb') as fp:
                pickle.dump(experiment, fp)
            
            
            print("Nb of fine runs:",len(likelihood_fine))
                                                 



def run_mcmc_pcn(problem, N, gamma, n_iter, reg, size_msh, theta_0, observations, covariates, scale_noise):
    
    # Create 'experiment' dictionary to keep track of runs and parameters
    experiment = {}
    experiment['problem'] = problem
    experiment['N'] = N
    experiment['gamma'] = gamma
    experiment['reg_type'] = reg['reg_type']
    experiment['size_mesh'] = size_msh
    experiment['theta_0'] = theta_0
    experiment['observations'] = observations
    experiment['covariates'] = covariates
    experiment['scale_noise'] = scale_noise
    
    # Create the filename in which to load the dictionary
    filename = f"experiment_PCN_{problem}_{reg['reg_type']}_{N}_{gamma}_{scale_noise}"
    
    
    # INITIALISATION
    theta = theta_0 + np.random.normal(loc=0.0, scale=0.1, size=K) #initialise close to true value

    # Compute the model and likelihood of this proposal on coarse mesh
    model_values = forward_map(problem, theta, size_msh, covariates)
    current_loglikelihood = compute_loglikelihood(model_values, observations, scale_noise)
    
    # Useful tables to store values
    chain = np.array([np.array(theta)])    
    likelihoods = np.array([current_loglikelihood])
    accept_reject = np.array([True])


    #--------------------ITERATE--------------------------------------------------
    for iter in range(n_iter): 
    
        
        # Get the proposal for new value
        proposal = compute_proposal_pcn(theta, gamma, reg)

        # Compute the model and likelihood of this proposal
        model_values = forward_map(problem, proposal, size_msh, covariates)
        proposal_loglikelihood = compute_loglikelihood(model_values, observations, scale_noise)

        # Metropolis step
        accept = metropolis_step(current_loglikelihood, proposal_loglikelihood, accept_reject)
        
        # Update the coarse tables
        if accept:
            accept_reject = np.append(accept_reject,[True])
            theta = proposal.copy()
            current_loglikelihood = proposal_loglikelihood

        else:
            accept_reject = np.append(accept_reject,[False])

        chain = np.append(chain, np.array([theta]),axis=0)
        likelihoods = np.append(likelihoods, [current_loglikelihood])
    
        
        # Save runs in dictionary
        experiment['chain'] = np.array(chain)
        experiment['likelihood'] = np.array(likelihoods)
        experiment['acceptance'] = np.array(accept_reject)

        with open(f'{filename}.pkl', 'wb') as fp:
            pickle.dump(experiment, fp)

        
                                       

