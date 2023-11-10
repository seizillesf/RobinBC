# Bayesian approach to inverse Robin problems

Scripts and notebooks to run and analyse the Bayesian computational routine to tackle the inverse problems of estimating a Robin boundary coefficient in Laplace Stokes PDE problems. Based on work by Mark Girolami, Ieva Kazlauskaite, Alsel K. Rasmussen and Fanny Seizilles.

## Set up
Included in this repository is the ```environment.yml``` file, that contains all the necessary packages that need to be installed in a conda environment in order to run our code. To setup the environment run:

    conda env create --name robin --file environment.yml
    conda activate robin

## Contents
In addition to the environment file, this repository contains 2 python files and a notebook.

- With the python file ```icesheet_functions.py```, all functions necessary to run the experiments are defined (PDE solver via ```FenicsX``` for FEM, MCMC)
- With the python file ```run.py```, we set the parameters of the experiment (prior type, number of observations...) and run it. For every experiment, a dictionary is created and stored in a ```.pkl``` file
- With the notebook ```Postprocessing.ipynb```, we define all the functions needed to analyse the results of the experiment once it has finished running. The ```.pkl``` file is retrieved and we can analyse the chains, compute the posterior mean and make reconstructions of the basal drag.

