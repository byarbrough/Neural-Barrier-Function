import sys

# Append the path to the directory containing your module to sys.path
import os

script_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_directory)

project_dir = os.path.dirname(script_directory)
sys.path.append(project_dir)

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch


if __name__ == '__main__':
    plot_opt = 'mip'  # 'mip' or 'bab'

    ################ load data from the double integrator example ################
    if plot_opt == 'mip':
        DI_result_dir = os.path.join(script_directory, 'double_integrator_mip', 'data')
    else:
        DI_result_dir = os.path.join(script_directory, 'double_integrator_bab', 'data')

    accpm_result = {
        'status': [],
        'timeout_status': [],
        'train_time': [],
        'verification_time': [],
        'runtime': [],
    }
    for i in range(8):
        accpm_result_path = os.path.join(
            DI_result_dir, 'train_iter_accpm_' + plot_opt + '_' + str(i) + '.p'
        )
        result = torch.load(accpm_result_path, map_location=torch.device('cpu'))
        accpm_result['status'].append(result['training_results']['status'])
        accpm_result['timeout_status'].append(
            result['training_results']['timeout_status']
        )
        accpm_result['train_time'].append(sum(result['training_results']['train_time']))
        accpm_result['verification_time'].append(
            sum(result['training_results']['verification_time'])
        )
        accpm_result['runtime'].append(
            sum(result['training_results']['train_time'])
            + sum(result['training_results']['verification_time'])
        )

    verify_result = {
        'status': [],
        'timeout_status': [],
        'train_time': [],
        'verification_time': [],
        'runtime': [],
    }
    for i in range(8):
        verify_result_path = os.path.join(
            DI_result_dir, 'train_iter_verify_' + plot_opt + '_' + str(i) + '.p'
        )
        result = torch.load(verify_result_path, map_location=torch.device('cpu'))
        verify_result['status'].append(result['training_results']['status'])
        verify_result['timeout_status'].append(
            result['training_results']['timeout_status']
        )
        verify_result['train_time'].append(
            sum(result['training_results']['train_time'])
        )
        verify_result['verification_time'].append(
            sum(result['training_results']['verification_time'])
        )
        verify_result['runtime'].append(
            sum(result['training_results']['train_time'])
            + sum(result['training_results']['verification_time'])
        )

    accpm_avg_runtime = [
        accpm_result['runtime'][i]
        for i in range(len(accpm_result['runtime']))
        if accpm_result['status'][i] == 'feasible'
    ]
    if len(accpm_avg_runtime) > 0:
        accpm_mip_avg_runtime = sum(accpm_avg_runtime) / len(accpm_avg_runtime)
    else:
        accpm_mip_avg_runtime = None

    verify_avg_runtime = [
        verify_result['runtime'][i]
        for i in range(len(verify_result['runtime']))
        if verify_result['status'][i] == 'feasible'
    ]
    if len(verify_avg_runtime) > 0:
        verify_mip_avg_runtime = sum(verify_avg_runtime) / len(verify_avg_runtime)
    else:
        verify_mip_avg_runtime = None

    accpm_DI_avg_runtime, verify_DI_avg_runtime = (
        accpm_mip_avg_runtime,
        verify_mip_avg_runtime,
    )
    print(
        'Double integrator average runtime: accpm: ',
        accpm_DI_avg_runtime,
        ' verify: ',
        verify_DI_avg_runtime,
    )

    ########### plot the runtime ###############
    fig = plt.subplots(figsize=(12, 8))
    num_exp = 8

    accpm_runtime = accpm_result['runtime']
    verify_runtime = verify_result['runtime']

    plt.scatter(range(num_exp), accpm_runtime, s=250, c='b', marker='o')
    plt.scatter(range(num_exp), verify_runtime, s=250, c='g', marker='o')

    # plot the time-out threshld 7200s
    plt.plot(
        range(num_exp), [7200] * num_exp, 'k--', label='Time out threshold', linewidth=3
    )

    ################ load data from the 6D quadrotor example ################
    if plot_opt == 'mip':
        quad_result_dir = os.path.join(script_directory, 'quadrotor_6D_mip', 'data')
    else:
        quad_result_dir = os.path.join(script_directory, 'quadrotor_6D_bab', 'data')

    accpm_result = {
        'status': [],
        'timeout_status': [],
        'train_time': [],
        'verification_time': [],
        'runtime': [],
    }
    for i in range(8):
        accpm_result_path = os.path.join(
            quad_result_dir, 'train_iter_accpm_' + plot_opt + '_' + str(i) + '.p'
        )
        result = torch.load(accpm_result_path, map_location=torch.device('cpu'))
        accpm_result['status'].append(result['training_results']['status'])
        accpm_result['timeout_status'].append(
            result['training_results']['timeout_status']
        )
        accpm_result['train_time'].append(sum(result['training_results']['train_time']))
        accpm_result['verification_time'].append(
            sum(result['training_results']['verification_time'])
        )
        accpm_result['runtime'].append(
            sum(result['training_results']['train_time'])
            + sum(result['training_results']['verification_time'])
        )

    verify_result = {
        'status': [],
        'timeout_status': [],
        'train_time': [],
        'verification_time': [],
        'runtime': [],
    }
    for i in range(8):
        verify_result_path = os.path.join(
            quad_result_dir, 'train_iter_verify_' + plot_opt + '_' + str(i) + '.p'
        )
        result = torch.load(verify_result_path, map_location=torch.device('cpu'))
        verify_result['status'].append(result['training_results']['status'])
        verify_result['timeout_status'].append(
            result['training_results']['timeout_status']
        )
        verify_result['train_time'].append(
            sum(result['training_results']['train_time'])
        )
        verify_result['verification_time'].append(
            sum(result['training_results']['verification_time'])
        )
        verify_result['runtime'].append(
            sum(result['training_results']['train_time'])
            + sum(result['training_results']['verification_time'])
        )

    accpm_avg_runtime = [
        accpm_result['runtime'][i]
        for i in range(len(accpm_result['runtime']))
        if accpm_result['status'][i] == 'feasible'
    ]
    if len(accpm_avg_runtime) > 0:
        accpm_mip_avg_runtime = sum(accpm_avg_runtime) / len(accpm_avg_runtime)
    else:
        accpm_mip_avg_runtime = None

    verify_avg_runtime = [
        verify_result['runtime'][i]
        for i in range(len(verify_result['runtime']))
        if verify_result['status'][i] == 'feasible'
    ]
    if len(verify_avg_runtime) > 0:
        verify_mip_avg_runtime = sum(verify_avg_runtime) / len(verify_avg_runtime)
    else:
        verify_mip_avg_runtime = None

    accpm_quad_avg_runtime, verify_quad_avg_runtime = (
        accpm_mip_avg_runtime,
        verify_mip_avg_runtime,
    )
    print(
        '6D quadrotor average runtime: accpm: ',
        accpm_quad_avg_runtime,
        ' verify: ',
        verify_quad_avg_runtime,
    )

    ########### plot the runtime ###############
    accpm_runtime = accpm_result['runtime']
    verify_runtime = verify_result['runtime']

    plt.scatter(range(num_exp), accpm_runtime, s=250, c='b', marker='^')
    plt.scatter(range(num_exp), verify_runtime, s=250, c='g', marker='^')

    plt.ylim([10, 5 * 1e4])
    plt.yticks(fontsize=22)
    plt.xticks(fontsize=22)

    # semilogy scale
    plt.yscale('log')
    plt.xlabel('Random seed', fontweight='bold', fontsize=24)
    plt.ylabel('Runtime in seconds', fontweight='bold', fontsize=24)

    plt.grid(True)

    plt.tight_layout(pad=1.0)

    circle = plt.Line2D(
        [0],
        [0],
        marker='o',
        color='black',
        markerfacecolor='none',
        markersize=20,
        linestyle='None',
    )
    triangle = plt.Line2D(
        [0],
        [0],
        marker='^',
        color='black',
        markerfacecolor='none',
        markersize=20,
        linestyle='None',
    )
    blue = plt.Line2D([0], [0], marker='s', color='b', markersize=20, linestyle='None')
    green = plt.Line2D([0], [0], marker='s', color='g', markersize=20, linestyle='None')

    proxy = [circle, triangle, blue, green]

    labels = ['Double integrator', '6D quadrotor', 'Fine-tuning', 'Verification-only']

    # Add legend to the plot
    plt.legend(proxy, labels, fontsize=20, loc='lower right', ncol=2)

    plt.savefig('runtime_' + plot_opt + '.png', dpi=300)

    plt.show()
