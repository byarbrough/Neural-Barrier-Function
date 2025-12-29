"""
BAB-based verification of neural network barrier functions.

Uses alpha-beta-CROWN's ABCrownSolver API for branch-and-bound verification.
"""

import copy
import time

import torch
import torch.nn as nn
import numpy as np

# Setup paths for alpha-beta-CROWN submodule
import setup_paths  # noqa: F401

# Import from alpha-beta-CROWN
from api import ABCrownSolver, VerificationSpec

# Import config with barrier-specific arguments
import config as arguments


class DecCondition(nn.Module):
    """NN module encoding the decreasing condition A@B(x) - B(x_next).

    For barrier function verification, we need to verify that
    A @ B(x) - B(f(x)) <= 0 for all x in workspace X.
    """

    def __init__(self, barrier_net, A_mat, dyn_net):
        super().__init__()
        device = A_mat.device
        n = A_mat.size(0)
        # Keep bias terms to make auto-lirpa work properly
        self.lin_layer = nn.Linear(n, n, bias=True).to(device)
        self.lin_layer.weight.data = A_mat
        self.lin_layer.bias.data = torch.zeros(n).to(device)

        self.barrier_net = barrier_net
        self.dyn_net = dyn_net

    def forward(self, x):
        return self.lin_layer(self.barrier_net(x)) - self.barrier_net(self.dyn_net(x))


def _build_config_dict(yaml_file_path=None):
    """Build configuration dictionary for ABCrownSolver."""
    # Get base config from arguments
    config = {}

    # General settings
    config["general"] = {
        "device": arguments.Config["general"]["device"],
        "seed": arguments.Config["general"]["seed"],
        "complete_verifier": "bab",
        "enable_incomplete_verification": True,
    }

    # BAB settings
    config["bab"] = {
        "timeout": arguments.Config["bab"]["timeout"],
        "branching": arguments.Config["bab"]["branching"],
    }

    # Solver settings
    config["solver"] = {
        "bound_prop_method": arguments.Config["solver"]["bound_prop_method"],
        "alpha-crown": arguments.Config["solver"]["alpha-crown"],
        "beta-crown": arguments.Config["solver"]["beta-crown"],
    }

    # Attack settings
    config["attack"] = {
        "pgd_order": arguments.Config["attack"]["pgd_order"],
        "pgd_steps": arguments.Config["attack"]["pgd_steps"],
        "pgd_restarts": arguments.Config["attack"]["pgd_restarts"],
    }

    return config


def bab_verification(nn_model, data_lb, data_ub, C, rhs, yaml_file_path=None):
    """
    Verify a neural network property using branch-and-bound.

    Args:
        nn_model: Neural network to verify
        data_lb: (B, nx) tensor - lower bounds of input domain
        data_ub: (B, nx) tensor - upper bounds of input domain
        C: (N, ny) tensor - output constraint matrix
        rhs: (N, 1) tensor - constraint RHS, defines C@y <= rhs
        yaml_file_path: Path to YAML config for BAB parameters

    Returns:
        verified_status: 'safe', 'unsafe', or 'unknown'
        bab_result: Dictionary with verification details
    """
    device = data_lb.device
    start_time = time.time()

    # Ensure proper shape
    if data_lb.dim() == 1:
        data_lb = data_lb.unsqueeze(0)
    if data_ub.dim() == 1:
        data_ub = data_ub.unsqueeze(0)

    # Build clauses for the specification
    # The spec format is: prove that NOT (C @ y <= rhs) is UNSAT
    # i.e., prove that C @ y > rhs never happens
    # But ABCrownSolver proves: output in safe region
    # So we need to flip: we want to prove C @ y <= rhs for all inputs
    # Actually, the unsafe region is C @ y <= rhs (want to find counterexample)
    # For barrier: we want to show B(x) > 0 is impossible in unsafe set
    # So spec clause is: (C, rhs) meaning find where C @ y <= rhs

    C_np = C.cpu().numpy() if torch.is_tensor(C) else C
    rhs_np = rhs.cpu().numpy() if torch.is_tensor(rhs) else rhs

    # Create output spec clauses - each row of C is an AND constraint
    # Format: list of (C_tensor, rhs_tensor) tuples for OR clauses
    # For barrier verification, each constraint is an independent check
    clauses = []
    for i in range(C_np.shape[0]):
        c_row = torch.tensor(C_np[i : i + 1], dtype=torch.float32)
        rhs_val = torch.tensor([rhs_np[i].item()], dtype=torch.float32)
        clauses.append((c_row, rhs_val))

    # Build verification spec
    spec = VerificationSpec.build_from_input_bounds(
        lower=data_lb.cpu(), upper=data_ub.cpu(), clauses=clauses
    )

    # Build config
    config = _build_config_dict(yaml_file_path)

    # Create solver and run verification
    solver = ABCrownSolver(
        spec=spec, computing_graph=nn_model, config=config, name="barrier_verification"
    )

    result = solver.solve()
    solver_time = time.time() - start_time

    # Parse result
    verified_status = result.status
    verified_success = result.success

    # Extract adversarial samples if any
    adv_samples = None
    if "counterexamples" in result.reference:
        adv_samples = result.reference["counterexamples"]
    elif "attack_examples" in result.reference:
        adv_samples = result.reference["attack_examples"]

    # Build result dictionary matching old interface
    bab_result = {
        "incomplete_lb": result.reference.get("global_lb", None),
        "bab_lb": result.reference.get("bab_lb", None),
        "bab_ub": result.reference.get("bab_ub", None),
        "status": verified_status,
        "status_lst": [verified_status],
        "solver_time": solver_time,
        "adv_samples": [[None, adv_samples]] if adv_samples is not None else None,
        "bab_sol": None,
        "bab_time_out": result.reference.get("timeout", False),
        "data_lb": data_lb,
        "data_ub": data_ub,
        "stats": result.stats,
    }

    return verified_status, bab_result


def bab_barrier_fcn_verification(problem, type, yaml_file_path, net=None):
    """
    Verify barrier function conditions using BAB.

    Args:
        problem: Problem object with barrier_fcn, system, X, X0, Xu
        type: 'xu' (unsafe), 'x0' (initial), or 'x' (decrease condition)
        yaml_file_path: Path to BAB YAML config
        net: Optional network to verify (uses problem.barrier_fcn.net if None)

    Returns:
        verified_status: 'safe', 'unsafe', or 'unknown'
        adv_samples: Numpy array of adversarial samples or None
        bab_result: Full result dictionary
    """
    system = problem.system
    B = problem.barrier_fcn
    device = problem.device

    if net is None:
        net = B.net

    if type == "xu":
        # Verify unsafe region condition: B(x) > 0 for all x in Xu
        # Find counterexample where B(x) <= tol (unsafe)
        Xu = problem.Xu
        assert Xu.is_box
        data_lb = torch.from_numpy(Xu.lb.astype("float32")).unsqueeze(0).to(device)
        data_ub = torch.from_numpy(Xu.ub.astype("float32")).unsqueeze(0).to(device)

        C = torch.zeros(1, B.n_out)
        C[0][B.unsafe_index] = 1.0
        tol_unsafe = arguments.Config["alg_options"]["barrier_fcn"]["train_options"][
            "condition_tol"
        ]
        rhs = torch.tensor([[tol_unsafe]])

        verified_status, bab_result = bab_verification(
            net, data_lb, data_ub, C, rhs, yaml_file_path
        )

        if bab_result["adv_samples"] is not None:
            adv_sample_filter_radius_xu = arguments.Config["alg_options"]["bab"][
                "adv_sample_filter_radius_xu"
            ]
            filtered_samples = [
                filter_adversarial_samples(
                    item[1], filter_radius=adv_sample_filter_radius_xu
                )
                for item in bab_result["adv_samples"]
                if item is not None
            ]
            filtered_samples = [item for item in filtered_samples if item is not None]
            if len(filtered_samples) > 0:
                adv_samples = torch.cat(filtered_samples, dim=0)
                adv_samples = adv_samples.detach().cpu().numpy()
            else:
                adv_samples = None
        else:
            adv_samples = None

        return verified_status, adv_samples, bab_result

    elif type == "x0":
        # Verify initial region condition: B(x) <= 0 for all x in X0
        # Find counterexample where -B(x) < -tol, i.e., B(x) > tol
        X0 = problem.X0
        assert X0.is_box
        data_lb = torch.from_numpy(X0.lb.astype("float32")).unsqueeze(0).to(device)
        data_ub = torch.from_numpy(X0.ub.astype("float32")).unsqueeze(0).to(device)

        C = -torch.eye(B.n_out)
        tol_init = arguments.Config["alg_options"]["barrier_fcn"]["train_options"][
            "condition_tol"
        ]
        rhs = tol_init * torch.ones(B.n_out, 1)

        verified_status, bab_result = bab_verification(
            net, data_lb, data_ub, C, rhs, yaml_file_path
        )

        adv_sample_filter_radius_x0 = arguments.Config["alg_options"]["bab"][
            "adv_sample_filter_radius_x0"
        ]
        if bab_result["adv_samples"] is not None:
            filtered_samples = [
                filter_adversarial_samples(
                    item[1], filter_radius=adv_sample_filter_radius_x0
                )
                for item in bab_result["adv_samples"]
                if item is not None
            ]
            filtered_samples = [item for item in filtered_samples if item is not None]
            if len(filtered_samples) > 0:
                adv_samples = torch.cat(filtered_samples, dim=0)
                adv_samples = adv_samples.detach().cpu().numpy()
            else:
                adv_samples = None
        else:
            adv_samples = None

        return verified_status, adv_samples, bab_result

    elif type == "x":
        # Verify decrease condition: A@B(x) - B(f(x)) <= 0 for all x in X
        X = problem.X
        assert X.is_box
        data_lb = torch.from_numpy(X.lb.astype("float32")).unsqueeze(0).to(device)
        data_ub = torch.from_numpy(X.ub.astype("float32")).unsqueeze(0).to(device)

        C = torch.eye(B.n_out)
        tol_dec = arguments.Config["alg_options"]["barrier_fcn"]["train_options"][
            "condition_tol"
        ]
        rhs = tol_dec * torch.ones(B.n_out, 1)

        barrier_net = copy.deepcopy(net)
        dyn_net = copy.deepcopy(system)
        A_mat = copy.deepcopy(B.A)

        dec_net = DecCondition(barrier_net, A_mat, dyn_net)

        verified_status, bab_result = bab_verification(
            dec_net, data_lb, data_ub, C, rhs, yaml_file_path
        )

        adv_sample_filter_radius_x = arguments.Config["alg_options"]["bab"][
            "adv_sample_filter_radius_x"
        ]
        if bab_result["adv_samples"] is not None:
            filtered_samples = [
                filter_adversarial_samples(
                    item[1], filter_radius=adv_sample_filter_radius_x
                )
                for item in bab_result["adv_samples"]
                if item is not None
            ]
            filtered_samples = [item for item in filtered_samples if item is not None]
            if len(filtered_samples) > 0:
                adv_samples = torch.cat(filtered_samples, dim=0)
                adv_samples = adv_samples.detach().cpu().numpy()
            else:
                adv_samples = None
        else:
            adv_samples = None

        return verified_status, adv_samples, bab_result

    else:
        raise ValueError(f"Unknown verification type: {type}")


def filter_adversarial_samples(samples, filter_radius=1e-2):
    """
    Remove duplicate adversarial samples within filter_radius distance.

    Args:
        samples: N x n tensor with most adversarial samples first
        filter_radius: Minimum distance between kept samples

    Returns:
        Filtered samples tensor or None
    """
    if samples is None:
        return samples

    numpy_input = False
    if isinstance(samples, np.ndarray):
        samples = torch.from_numpy(samples)
        numpy_input = True

    if samples.dim() == 0 or len(samples) == 0:
        return None

    new_samples = samples[0].unsqueeze(0)
    filtered_sample = samples[0]

    while True:
        index = torch.sum((samples - filtered_sample) ** 2, dim=1) > filter_radius**2
        samples = samples[index]
        if len(samples) == 0:
            break

        filtered_sample = samples[0]
        new_samples = torch.cat((new_samples, filtered_sample.unsqueeze(0)), dim=0)

    if numpy_input:
        new_samples = new_samples.cpu().numpy()

    return new_samples
