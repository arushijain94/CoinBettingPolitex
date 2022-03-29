# Towards Painless Policy Optimization for Constrained MDPs

We run CMDP with Coin-Betting algorithm (Orabona and Tommasi, 2017) and compare performance against two baselines GDA (Ding et al., 2020) and CRPO (Xu et al., 2021).
See the following directory to run different types of algorithm.

* **Tabular**: Code for Gridworld environment.
* **Cartpole**: Code for Cartpole environment.

## Installation
* Create a virtual env using python3

`virtualenv -p python3 <envname>`

* Activate virtual environment

`source ./envname/bin/activate`

* We use easy MDPs implemented with gym like interface using 
emdp library <https://github.com/zafarali/emdp>. To install emdp:

`cd` into `emdp` directory and run:
`pip install -e .`
* Install other libraries using `requirements.txt` file

`pip install -r requirements.txt`

## How to run code?

### Tabular Gridworld Environment

#### Model-based
Here, we assume that model (transition and reward) of environment is known.
* Folder **ModelBased**

`CBP.py`, `GDA.py`, `CRPO.py` contains the code for running CBP, and baselines - GDA and CRPO.
To run the code on cluster use `bash run_slurm_CB.sh`, `bash run_slurm_GDA.sh`, `bash run_slurm_CRPO.sh`. Here we experiment with different parameter values.
See the paper for the optimal hyperparameters used.

`Plots/best_hyperparams.py` contain the plotting script to plot Optimality gap and Constraint violation for diff hyperparameters .
`Plots/model_based_sensitivity_gam.py` contain the plotting script to plot Optimality gap and Constraint violation for environment misspecification scenario .

#### Model-free
Next, we assume the case where model is not known beforehand. We assume a model-free setting where we use TD sampling approach to learn the Q value estimates.
* Folder **ModelFree**

`CBP_TDSampling.py`, `GDA_TDSampling.py`, `CRPO_TDSampling.py` contains the code for running CBP, and baselines - GDA and CRPO.
To run the code on cluster use `bash run_slurm_CB.sh`, `bash run_slurm_GDA.sh`, `bash run_slurm_CRPO.sh`. Here we experiment with different parameter values.
See the paper for the optimal hyperparameters used.

`Plots/model_free_tdsampling.py` contain the plotting script to plot Optimality gap and Constraint violation for diff number of samples to estimate Q values.

#### Linear Function Approximation
Next, we look into Linear function approximation (LFA) setting for gridworld environment. We use tile coding to learn the (s,a) features of the environment.
We use LSTD to learn the estimates of Q value functions for both reward and cost function.
* Folder **LinearFuncApprox**

`CBP_LSTD.py`, `GDA_LSTD.py`, `CRPO_LSTD.py` contains the code for running CBP, and baselines - GDA and CRPO.
To run the code on cluster use `bash run_slurm_CB_LSTD.sh`, `bash run_slurm_GDA_LSTD.sh`, `bash run_slurm_CRPO_LSTD.sh`. Here we experiment with different parameter values.
See the paper for the optimal hyperparameters used.

`Plots/model_free_tilecoding_feature_vary.py` contain the plotting script to plot Optimality gap and Constraint violation for diff number of feature dimension.

#### G-Experimental Design to build Coreset C
We now look into G-experimental design using Coreset formulation. Look for Appendix c.2 for further details on how to build a coreset.
* Folder **KW_Coreset**

First build the coreset states by running `python KWCoreset.py`. This will store the coresets which will be used further by algorithms below. 
`CBP_KW.py`, `GDA_KW.py`, `CRPO_KW.py` contains the code for running CBP, and baselines - GDA and CRPO.
To run the code on cluster use `bash run_slurm_CB_KW.sh`, `bash run_slurm_GDA_KW.sh`, `bash run_slurm_CRPO_KW.sh`. Here we experiment with different parameter values.
See the paper for the optimal hyperparameters used.

### Cartpole Environment
We experiment with Cartpole environment. The `Cartpole/Cartpole.py` contains the modified cartpole environment.
We added two constraint rewards (c1 , c2 ) to the classic OpenAI gym Cartpole environment.
(1) Cart receives a c1 = 0 constraint reward value when enters the area [−2.4, −2.2], [−1.3, −1.1], [1.1, 1.3], [2.2, 2.4], else receive c1 = +1.
(2) When the angle of the cart is less than 4 degrees receive c2 = +1, else everywhere c2 = 0. Each episode length is no longer than 200.

We use tile coding (Sutton and Barto, 2018) to discretize continuous state space of env.
* Folder **Cartpole**

First run `python SlaterConstant.py` to get upper bound of lambda variable for Cartpole ennvironment. We use Coin-Betting Algorithm to learn upper bound on lambda.  
The files: `trainCartpole_CBP.py`, `trainCartpole_GDA.py`, `trainCartpole_CRPO.py` contains the code for running CBP, and baselines - GDA and CRPO.
To run the code on cluster use `bash run_slurm_CB.sh`, `bash run_slurm_GDA.sh`, `bash run_slurm_CRPO.sh`. Here we experiment with different parameter values.
See the paper for the optimal hyperparameters used.

`Plots/best_hyperparameter.py` contains the plotting script to plot Optimality gap and Constraint violation for best hyperparameters in dark shade and other hyperparameter values in lighter shade.




