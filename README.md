# Verification-Aided Learning of NN Vector Barrier Functions 
Verifying the safety of a complex dynamical system is a long-standing challenge due to the computational complexity and the need of domain knowledge. Can we automate the safety verification by learning a barrier function that separates the system's trajectory and the unsafe regions, while maintaining the formal guarantee of the learned certificate? 

The answer is affirmative, and calls for a fusion of learning and verification tools. How to get the best from both learning and verification is the main problem considered in the paper:

- [Verification-Aided Learning of Neural Network Barrier Functions with
Termination Guarantees](https://arxiv.org/pdf/2403.07308). Shaoru Chen, Lekan Molu, Mahyar Fazlyab. American Control Conference (ACC), 2024

More specifically, we want to find the best way to (1) handle verification failure and (2) retrain the NN safety certificate. 

- A straightforward method follows the "train-verify" procedure: we first learn a barrier function candidate and verify if it is valid. If the verification fails, we augment the training dataset with the counterexamples found and retrain the NN barrier function. However, we don't have much control over this process and the loop between training and verification may never terminate.
- This repository implements a "**train-finetune**" procedure where the last linear layer of the NN is fine-tuned through **a convex optimization algorithm** that achieves strong convergence guarantees. The "train-finetune" framework also achieves boosted success rate in learning a formal NN safety certificate in practice. 

<img src="https://github.com/ShaoruChen/Neural-Barrier-Function/blob/main/Documents/Figures/VAL_problem_overview.png" width=500, height=300> 

## Verification-Aided Learning Framework 
Intuitively, we want to rely on machine learning to find a candidate solution that is close to being valid and close the final gap using optimization. The verification-aided learning framework consists of three main steps:

1. Train a NN vector barrier function
2. Fine-tune the **last linear layer** of the NN barrier function to achieve a barrier function with formal guarantee
3. In case of search failure, generate counterexamples to augment the dataset and retrain the model using sequential learning techniques

The fine-tuning step follows a counterexample guided inductive synthesis (CEGIS) framework and can be considered as applying cutting-plane algorithms in the parameter space of the last layer's weights. A detailed explanation of the fine-tuning step can be found in the paper. 

<img src="https://github.com/ShaoruChen/Neural-Barrier-Function/blob/main/Documents/Figures/VAL_fine_tuning_diagram.png" width=600, height=300> 

## Numerical Examples
For a double integrator system controlled by a ReLU network controller, the following figure shows that the learned NN barrier function can effectively separate the system's trajectories and the unsafe region:

<img src="https://github.com/ShaoruChen/Neural-Barrier-Function/blob/main/Documents/Figures/VAL_DI_barrier.png" width=400, height=300> 

The following table shows that the proposed "train-finetune" framework achieves better success rate than the "train-verify" framework in learning a valid NN barrier function. 

<img src="https://github.com/ShaoruChen/Neural-Barrier-Function/blob/main/Documents/Figures/VAL_success_rate.png" width=500, height=200> 

## Codebase Organization 
- In the experiments, we apply a mixed-integer programming (MIP)-based NN verifier and [alpha, beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN) for testing. The implementation of the MIP-based verificaiton is based on [Gurobi](https://www.gurobi.com/resources/mixed-integer-programming-mip-a-primer-on-the-basics/) and can be found in [ACCPM.py](https://github.com/ShaoruChen/Neural-Barrier-Function/blob/150067fa3d4db9db8ecfc62bbebd696714213281/cutting_plane/ACCPM.py#L137). Codes of alpha, beta-CROWN is included in the [complete_verifier](https://github.com/ShaoruChen/Neural-Barrier-Function/tree/main/complete_verifier) folder where [complete_verification.py](https://github.com/ShaoruChen/Neural-Barrier-Function/blob/main/complete_verifier/complete_verification.py) provides interfaces for verification of NN barrier functions.
- Training of the NN barrier function and augmentation of the counterexamples are implemented in [train_barrier.py](https://github.com/ShaoruChen/Neural-Barrier-Function/blob/main/training/train_barrier.py).
- The fine-tuning algorithm is based on the analytic center cutting-plane method and is implemented in [ACCPM.py](https://github.com/ShaoruChen/Neural-Barrier-Function/blob/main/cutting_plane/ACCPM.py).


## Installation
The following environment setup has been tested. 

```bash
conda create --name test python==3.7.13
conda activate test

% install pytorch with a suitable version on your device, see (https://pytorch.org/get-started/previous-versions/). The following versions have been tested and recommended for use.

% CPU only
pip install torch==1.11.0+cpu torchvision==0.12.0+cpu torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cpu

or 

% CUDA 11.3
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113


% install packages
pip install -r requirements.txt
conda install -c conda-forge cvxpy
```

## Run experiments
To run the experiments in the `examples` folder, run the following commands in the terminal: 

```
cd examples/double_integrator_mip
python main_train_DI.py --config double_integrator.yaml --seed 0 --method fine-tuning
```

The experimental setup parameters are included in the `yaml` file with comments. 
