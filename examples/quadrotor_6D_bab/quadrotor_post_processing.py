import sys

# Append the path to the directory containing your module to sys.path
import os

script_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_directory)

project_dir = os.path.dirname(script_directory)
project_dir = os.path.dirname(project_dir)
sys.path.append(project_dir)

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch


if __name__ == '__main__':
    result_dir = os.path.join(script_directory, 'data')
    accpm_mip_result = {
        'status': [],
        'timeout_status': [],
        'train_time': [],
        'verification_time': [],
        'runtime': [],
    }
    for i in range(8):
        accpm_result_path = os.path.join(
            result_dir, 'train_iter_accpm_' + str(i) + '.p'
        )
        result = torch.load(accpm_result_path)
        accpm_mip_result['status'].append(result['training_results']['status'])
        accpm_mip_result['timeout_status'].append(
            result['training_results']['timeout_status']
        )
        accpm_mip_result['train_time'].append(
            sum(result['training_results']['train_time'])
        )
        accpm_mip_result['verification_time'].append(
            sum(result['training_results']['verification_time'])
        )
        accpm_mip_result['runtime'].append(
            sum(result['training_results']['train_time'])
            + sum(result['training_results']['verification_time'])
        )

    verify_mip_result = {
        'status': [],
        'timeout_status': [],
        'train_time': [],
        'verification_time': [],
        'runtime': [],
    }
    for i in range(8):
        verify_result_path = os.path.join(
            result_dir, 'train_iter_verify_' + str(i) + '.p'
        )
        result = torch.load(verify_result_path)
        verify_mip_result['status'].append(result['training_results']['status'])
        verify_mip_result['timeout_status'].append(
            result['training_results']['timeout_status']
        )
        verify_mip_result['train_time'].append(
            sum(result['training_results']['train_time'])
        )
        verify_mip_result['verification_time'].append(
            sum(result['training_results']['verification_time'])
        )
        verify_mip_result['runtime'].append(
            sum(result['training_results']['train_time'])
            + sum(result['training_results']['verification_time'])
        )

    accpm_mip_avg_runtime = [
        accpm_mip_result['runtime'][i]
        for i in range(len(accpm_mip_result['runtime']))
        if accpm_mip_result['status'][i] == 'feasible'
    ]
    if len(accpm_mip_avg_runtime) > 0:
        accpm_mip_avg_runtime = sum(accpm_mip_avg_runtime) / len(accpm_mip_avg_runtime)
    else:
        accpm_mip_avg_runtime = None

    verify_mip_avg_runtime = [
        verify_mip_result['runtime'][i]
        for i in range(len(verify_mip_result['runtime']))
        if verify_mip_result['status'][i] == 'feasible'
    ]
    if len(verify_mip_avg_runtime) > 0:
        verify_mip_avg_runtime = sum(verify_mip_avg_runtime) / len(
            verify_mip_avg_runtime
        )
    else:
        verify_mip_avg_runtime = None
