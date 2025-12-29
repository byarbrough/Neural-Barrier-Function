import sys
import os

script_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_directory)

project_dir = os.path.dirname(script_directory)
project_dir = os.path.dirname(project_dir)
sys.path.append(project_dir)

import torch
import torch.nn as nn
import numpy as np

from utils.sampling import scale_polyhedron

from dynamics.models import (
    Barrier_Fcn,
    Polyhedral_Set,
    OpenLoopDynamics,
    ClosedLoopDynamics,
    Controller,
)

from pympc.geometry.polyhedron import Polyhedron
from matplotlib.patches import Patch

import os

import json
import matplotlib.pyplot as plt

from dynamics.models import NeuralNetwork

if __name__ == "__main__":
    data_dir = "data"
    #################### initialize the verification problem ##############################
    x_dim = 2
    u_dim = 1
    # state space
    x_lb = np.array([-3.0, -3.0]).astype("float32")
    x_ub = np.array([3.0, 3.0]).astype("float32")
    domain = Polyhedron.from_bounds(x_lb, x_ub)
    domain = scale_polyhedron(domain, 1.0)

    X = Polyhedral_Set(domain.A, domain.b)
    X.lb, X.ub = x_lb, x_ub
    X.is_box = True

    # the initial region
    x0_min = np.array([2.5, -0.5]).astype("float32")
    x0_max = np.array([2.8, 0.0]).astype("float32")
    X0 = Polyhedron.from_bounds(x0_min, x0_max)
    X0 = Polyhedral_Set(X0.A, X0.b)
    X0.lb, X0.ub = x0_min, x0_max
    X0.is_box = True

    # the unsafe region
    xu_min = np.array([1.5, -0.4]).astype("float32")
    xu_max = np.array([2.1, 0.2]).astype("float32")

    Xu = Polyhedron.from_bounds(xu_min, xu_max)
    Xu = Polyhedral_Set(Xu.A, Xu.b)
    Xu.lb, Xu.ub = xu_min, xu_max
    Xu.is_box = True

    #################### generate autonomous nn dynamics ##############################
    # load dynamics model
    model_config_path = "model_data/doubleIntegrator.json"
    with open(model_config_path, "r") as file:
        config = json.load(file)

    device = torch.device("cpu")
    A = torch.Tensor(config["A"]).to(device)
    B = torch.Tensor(config["B"]).to(device)
    c = torch.Tensor(config["c"]).to(device)

    nn_controller_path = "model_data/doubleIntegratorCorrect.pth"
    DI_dyn = NeuralNetwork(nn_controller_path, A, B, c)
    DI_dyn = DI_dyn.to(device)

    # create open loop dynamics
    u_lb = np.array([-2.0]).astype("float32")
    u_ub = np.array([2.0]).astype("float32")
    aug_x_lb, aug_x_ub = np.concatenate((x_lb, u_lb)), np.concatenate((x_ub, u_ub))
    aug_x_domain = Polyhedron.from_bounds(aug_x_lb, aug_x_ub)

    open_loop_dyn_layer = nn.Sequential(nn.Linear(x_dim + u_dim, x_dim)).to(device)
    open_loop_dyn_layer[0].weight.data = torch.cat((A, B), dim=-1).to(device)

    # Note that the bias term c provided has the wrong dimension and needs to be reshaped
    open_loop_dyn_layer[0].bias.data = torch.zeros(x_dim).to(device)
    open_loop_dyn = OpenLoopDynamics(open_loop_dyn_layer, x_dim, u_dim, aug_x_domain)

    controller_net = DI_dyn.Linear
    controller = Controller(controller_net, domain)

    dynamics = ClosedLoopDynamics(open_loop_dyn, controller)

    #################### train a barrier function ##############################
    barrier_dim = 6
    barrier_net = nn.Sequential(
        nn.Linear(x_dim, 50),
        nn.ReLU(),
        nn.Linear(50, 30),
        nn.ReLU(),
        nn.Linear(30, 10),
        nn.ReLU(),
        nn.Linear(10, barrier_dim),
    )

    barrier_fcn = Barrier_Fcn(barrier_net, domain).to(device)

    # choose a verified NN CBF (index set of success examples: [0, 1, 3, 4, 5])
    result_path = os.path.join(script_directory, data_dir, "B_iter_model_accpm_mip_0.p")

    # choose a verified NN CBF (index set of success examples: [0, 5, 6])
    # result_path = os.path.join(script_directory, data_dir, 'B_iter_model_verify_mip_6.p')

    result = torch.load(result_path)

    barrier_fcn.load_state_dict(result)

    # plot
    fig, ax = plt.subplots(figsize=(12, 8))

    init_states = X0.sample(1000)
    init_states = torch.from_numpy(init_states).to(device)
    num_step = 10
    traj = dynamics.simulate_trajectory(init_states, num_step)

    traj = traj.detach().numpy()
    for i in range(num_step + 1):
        plt.scatter(traj[:, i, 0], traj[:, i, 1], marker=".", alpha=0.5)

    Xu.set.plot(fill=False, ec="r", linestyle="-", linewidth=3)
    X0.set.plot(fill=False, ec="b", linestyle="-", linewidth=3)
    X.set.plot(fill=False, ec="k", linestyle="-", linewidth=3)

    plot_x_range = [-3.3, 3.3]
    plot_y_range = [-3.3, 3.3]

    x_range = np.arange(plot_x_range[0], plot_x_range[1], 0.3).astype("float32")
    y_range = np.arange(plot_y_range[0], plot_y_range[1], 0.3).astype("float32")
    xx, yy = np.meshgrid(x_range, y_range)

    U = np.zeros(xx.shape)
    V = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            input = torch.tensor([[xx[i, j], yy[i, j]]])
            U[i, j] = (
                dynamics(input).detach().cpu().numpy()[0, 0].astype("float32")
                - xx[i, j]
            )
            V[i, j] = (
                dynamics(input).detach().cpu().numpy()[0, 1].astype("float32")
                - yy[i, j]
            )

    # plot vector fields
    q = ax.quiver(xx, yy, U, V, color="gray", alpha=0.7, headwidth=4, headlength=6)

    ind = 0

    def barrier_fcn_ind_output(x):
        return barrier_fcn(x)[:, ind]

    plot_x_range = [-3.0, 3.1]
    plot_y_range = [-3.0, 3.1]

    x_range = np.arange(plot_x_range[0], plot_x_range[1], 0.1).astype("float32")
    y_range = np.arange(plot_y_range[0], plot_y_range[1], 0.1).astype("float32")
    xx, yy = np.meshgrid(x_range, y_range)

    zz = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            input = torch.tensor([[xx[i, j], yy[i, j]]])
            zz[i, j] = (
                barrier_fcn_ind_output(input)
                .detach()
                .cpu()
                .numpy()[0]
                .astype("float32")
            )

    level_value = 0.0
    h = plt.contour(
        xx, yy, zz, [level_value], linewidths=3.0, colors="black", linestyles="dashed"
    )
    # plt.clabel(h, inline=True, fontsize=14, fmt='%1.1f')  # Label the contour lines

    plt.xlabel(r"$x_1$", fontweight="bold", fontsize=24)
    plt.ylabel(r"$x_2$", fontweight="bold", fontsize=24)

    plt.yticks(fontsize=22)
    plt.xticks(fontsize=22)
    plt.xlim([-3.1, 3.1])
    plt.ylim([-3.1, 3.1])

    proxy = [
        Patch(fill=False, linewidth=3.0, edgecolor="k"),
        Patch(fill=False, linewidth=3.0, edgecolor="r"),
        Patch(fill=False, linewidth=3.0, edgecolor="b"),
        plt.Line2D([0], [0], linestyle="--", color="k", linewidth=3.0),
    ]

    labels = [r"$\mathcal{X}$", r"$\mathcal{X}_u$", r"$\mathcal{X}_0$", r"$B_0(x) = 0$"]
    # Add legend to the plot
    plt.legend(proxy, labels, fontsize=20, loc="upper right")
    plt.tight_layout(pad=1.0)
    plt.savefig("barrier_level_set.png", dpi=300)
    plt.show()
