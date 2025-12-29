# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements verification-aided learning of neural network (NN) vector barrier functions for safety verification of dynamical systems. The key innovation is a "train-finetune" framework that combines machine learning with formal verification using cutting-plane algorithms.

### Core Methodology

1. **Train** a NN vector barrier function candidate
2. **Fine-tune** the last linear layer via convex optimization (ACCPM - Analytic Center Cutting-Plane Method)
3. **Verify** using either MIP (Mixed-Integer Programming) or alpha-beta-CROWN
4. If verification fails, augment dataset with counterexamples and iterate

The fine-tuning step achieves strong convergence guarantees and higher success rates compared to naive train-verify loops.

## Development Commands

### Environment Setup

```bash
# Clone with submodules
git clone --recursive <repo-url>
# Or if already cloned:
git submodule update --init --recursive

# Create environment with Python 3.11
conda env create -f environment.yml
conda activate neural-barrier

# Or manually:
conda create -n neural-barrier python=3.11
conda activate neural-barrier
pip install -r requirements.txt
pip install cvxpy

# Install auto-LiRPA from submodule
cd alpha-beta-CROWN/auto_LiRPA
pip install -e .
cd ../..
```

### Running Experiments

```bash
# Basic experiment run (from examples subdirectory)
cd examples/double_integrator_mip
python main_train_DI_mip.py --config double_integrator.yaml --seed 0 --method fine-tuning

# For BAB verification variant
cd examples/double_integrator_bab
python main_train_DI_bab.py --config double_integrator.yaml --seed 0 --method fine-tuning

# Quadrotor 6D examples
cd examples/quadrotor_6D_mip
python main_train_quad_mip.py --config quadrotor_6D.yaml --seed 0 --method fine-tuning
```

## Architecture

### Top-Level Directory Structure

- **alpha-beta-CROWN/**: Git submodule of Verified-Intelligence/alpha-beta-CROWN verifier
- **config/**: Configuration module extending alpha-beta-CROWN with barrier-specific arguments
- **cutting_plane/**: ACCPM algorithm for fine-tuning the last linear layer
- **dynamics/**: Dynamical system models and NN dynamics
- **training/**: Training logic for barrier functions
- **verification/**: Verification backends (MIP and BAB)
- **barrier_utils/**: Sampling, model transformations, training utilities (named to avoid conflict with alpha-beta-CROWN's utils module)
- **pympc/**: MPC utilities (control, dynamics, geometry, optimization)
- **examples/**: Runnable experiments (double integrator, quadrotor)
- **setup_paths.py**: Path configuration for imports

### Key Modules and Their Roles

#### Barrier Function Training (`training/train_barrier.py`)

- `Trainer`: Main class coordinating training, fine-tuning, and verification
- Manages sample sets for initial conditions (X0), unsafe regions (Xu), and workspace (X)
- Integrates with ACCPM for fine-tuning and verification backends
- Handles counterexample-guided data augmentation

#### Fine-Tuning Algorithm (`cutting_plane/ACCPM.py`)

- `ACCPM_Alg`: Implements analytic center cutting-plane method
- `Learner`: Solves the analytic center problem using cvxpy
- `Problem`: Encapsulates system, barrier function, and sets (X, X0, Xu)
- Fine-tunes only the last linear layer weights while keeping basis network fixed

#### Verification Backends

**MIP Verification** (`verification/mip_verification.py`):
- Uses Gurobi to encode NN constraints as MIP
- Provides exact verification with configurable time limits
- Returns counterexamples when verification fails

**BAB Verification** (`verification/bab_verification.py`):
- Wraps alpha-beta-CROWN complete verifier
- `DecCondition`: NN module encoding decrease condition A@B(x) - B(x_next)
- Supports PGD attack, incomplete verification (alpha-CROWN), and branch-and-bound
- Configuration via YAML files in example directories

#### Dynamical Systems (`dynamics/systems.py`, `dynamics/models.py`)

- `NN_Dynamics`: Neural network dynamics x_next = f(x)
- `Barrier_Fcn`: Vector barrier function with learnable matrix A
- `Polyhedral_Set`: Geometric sets for workspace, initial, and unsafe regions
- Pre-activation bound computation for verification

### Configuration System

All experiments use YAML configuration files (see `examples/*/double_integrator.yaml` or `quadrotor_6D.yaml`) with hierarchical structure:

```yaml
general:
  device: cpu/cuda
  seed: 0

alg_options:
  train_method: fine-tuning  # or 'verification-only'
  verification_method: mip   # or 'bab'
  barrier_fcn:
    barrier_output_dim: 6
    dataset: {...}
    train_options: {...}
  ACCPM:
    max_iter: 20
    cvxpy_solver: MOSEK
  mip: {...}
  bab: {...}
```

Configuration is parsed via `config.Config.parse_config()` (extends alpha-beta-CROWN's ConfigHandler).

### Important Implementation Details

1. **Barrier Function Structure**: All barrier functions follow a basis network + last linear layer architecture. Only the last layer is fine-tuned by ACCPM.

2. **Verification Methods**:
   - **MIP**: Exact but slower; good for small networks
   - **BAB (alpha-beta-CROWN)**: Scalable but may timeout; good for larger networks
   - Both return counterexamples on verification failure

3. **Pre-activation Bounds**: Both verification methods require pre-activation bounds for ReLU layers, computed via auto-LiRPA in `verification/nn_verification.py`.

4. **Sample Management**: Trainer maintains pool sizes (`samples_pool_size`) to prevent unbounded memory growth during iterative refinement.

5. **Device Handling**: Models and data are moved to device (CPU/CUDA) specified in config. Check `device` property of Problem, Barrier_Fcn, etc.

## Common Workflows

### Adding a New Dynamical System

1. Define system in `dynamics/systems.py` or load pre-trained NN controller
2. Create experiment directory in `examples/`
3. Define workspace (X), initial set (X0), unsafe set (Xu) as Polyhedron objects
4. Create YAML config specifying barrier function dimensions and verification parameters
5. Write main training script following pattern in `examples/double_integrator_mip/main_train_DI_mip.py`

### Debugging Verification Failures

- Check `ACCPM_result.p` pickle file for counterexamples
- Inspect `adv_samples` field in BAB results
- Visualize barrier function level sets using post-processing scripts in example directories
- Adjust `condition_tol` in YAML config to relax barrier conditions
- Increase `num_ce_samples` to generate more samples around counterexamples

### Tuning Verification Performance

**For MIP**:
- Set `time_limit` (0.0 = no limit)
- Adjust `MIPFocus` (0-3, see Gurobi docs)

**For BAB**:
- Increase `timeout` for harder problems
- Tune `branching.method` (babsr=fast, kfsb=balanced, fsb=slow/accurate)
- Adjust `alpha-crown.iteration` and `beta-crown.iteration`
- Increase `pgd_steps` and `pgd_restarts` for stronger adversarial filtering

## Dependencies and External Tools

- **PyTorch 2.4+**: Neural network implementation and training
- **Python 3.11**: Required for compatibility with latest alpha-beta-CROWN
- **alpha-beta-CROWN**: Branch-and-bound verification (included as git submodule)
- **auto-LiRPA**: Bound propagation for verification (included in alpha-beta-CROWN)
- **cvxpy**: Convex optimization for ACCPM
- **Gurobi** (optional): MIP solver for MIP verification method (requires license)
- **MOSEK** (optional): Alternative cvxpy solver, can be faster than default

The `alpha-beta-CROWN/` directory is a git submodule of the upstream repository. The `config/` module extends its configuration with barrier-specific arguments.

## Paper Reference

Implementation based on:
"Verification-Aided Learning of Neural Network Barrier Functions with Termination Guarantees"
Shaoru Chen, Lekan Molu, Mahyar Fazlyab. ACC 2024
https://arxiv.org/pdf/2403.07308
