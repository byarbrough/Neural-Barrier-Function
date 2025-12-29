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
    # load mip verification result
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
            result_dir, 'train_iter_accpm_mip_' + str(i) + '.p'
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
            result_dir, 'train_iter_verify_mip_' + str(i) + '.p'
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

    # load bab verification result
    # result_dir = '/Users/shaoruchen/Documents/GitHub/learning-basis-functions/codes/learning_barrier/examples/double_integrator/data_bab'
    # accpm_bab_result = {'status': [], 'timeout_status': [], 'train_time': [], 'verification_time': [], 'runtime': []}
    # for i in range(5):
    #     accpm_result_path = os.path.join(result_dir, 'train_iter_accpm_bab_' + str(i) + '.p')
    #     result = torch.load(accpm_result_path)
    #     accpm_bab_result['status'].append(result['training_results']['status'])
    #     accpm_bab_result['timeout_status'].append(result['training_results']['timeout_status'])
    #     accpm_bab_result['train_time'].append(sum(result['training_results']['train_time']))
    #     accpm_bab_result['verification_time'].append(sum(result['training_results']['verification_time']))
    #     accpm_bab_result['runtime'].append(
    #         sum(result['training_results']['train_time']) + sum(result['training_results']['verification_time']))
    #
    # verify_bab_result = {'status': [], 'timeout_status': [], 'train_time': [], 'verification_time': [], 'runtime': []}
    # for i in range(5):
    #     verify_result_path = os.path.join(result_dir, 'train_iter_verify_bab_' + str(i) + '.p')
    #     result = torch.load(verify_result_path)
    #     verify_bab_result['status'].append(result['training_results']['status'])
    #     verify_bab_result['timeout_status'].append(result['training_results']['timeout_status'])
    #     verify_bab_result['train_time'].append(sum(result['training_results']['train_time']))
    #     verify_bab_result['verification_time'].append(sum(result['training_results']['verification_time']))
    #     verify_bab_result['runtime'].append(
    #         sum(result['training_results']['train_time']) + sum(result['training_results']['verification_time']))

    accpm_bab_result = {
        'status': [
            'feasible',
            'feasible',
            'feasible',
            'feasible',
            'feasible',
            'unknown',
            'feasible',
            'feasible',
        ],
        'train_time': [
            711.5414588451385,
            262.55511355400085,
            232.2060570716858,
            88.2577908039093,
            223.66415286064148,
            1685.1845207214355,
            541.0353586673737,
            400.7866668701172,
        ],
        'verification_time': [
            1562.2223294417636,
            866.7566264043497,
            622.3885173946541,
            312.2085938547513,
            335.3840737237357,
            5095.144294822882,
            1380.1863374963573,
            474.49306030166065,
        ],
        'runtime': [
            2273.763788286902,
            1129.3117399583507,
            854.5945744663399,
            400.4663846586606,
            559.0482265843772,
            6780.328815544318,
            1921.221696163731,
            875.2797271717778,
        ],
    }

    verify_bab_result = {
        'status': [
            'unknown',
            'unknown',
            'unknown',
            'unknown',
            'unknown',
            'unknown',
            'unknown',
            'unknown',
        ],
        'train_time': [
            3562.8503324985504,
            2275.0539939403534,
            3695.582419395447,
            3509.6120920181274,
            3641.807368993759,
            3322.541230916977,
            3264.162317752838,
            3726.0518136024475,
        ],
        'verification_time': [
            3509.45818734169,
            2099.8580067157745,
            3633.4932174682617,
            3499.6162028312683,
            3361.6554436683655,
            3740.125264406204,
            3845.498171567917,
            3311.067379951477,
        ],
        'runtime': [
            7072.3085198402405,
            4374.912000656128,
            7329.0756368637085,
            7009.228294849396,
            7003.462812662125,
            7062.666495323181,
            7109.660489320755,
            7037.119193553925,
        ],
    }

    num_exp = 8

    option = 'mip'
    # plot
    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))

    # Set position of bar on X axis
    br1 = np.arange(num_exp)
    br2 = [x + barWidth for x in br1]

    if option == 'mip':
        # Make the plot
        accpm_bars = plt.bar(
            br1,
            accpm_mip_result['runtime'],
            width=barWidth,
            hatch=['', '', '', '', '', '', '', ''],
            edgecolor='black',
            label='Fine-tuning',
        )

        verify_bars = plt.bar(
            br2,
            verify_mip_result['runtime'],
            width=barWidth,
            hatch=['', '', 'xx', 'xx', '', '', '', ''],
            edgecolor='black',
            label='Verification-only',
        )
    else:
        accpm_bars = plt.bar(
            br1,
            accpm_bab_result['runtime'],
            width=barWidth,
            hatch=['', '', '', '', '', 'xx', '', ''],
            edgecolor='black',
            label='Fine-tuning',
        )

        verify_bars = plt.bar(
            br2,
            verify_bab_result['runtime'],
            width=barWidth,
            hatch=['xx', 'xx', 'xx', 'xx', 'xx', 'xx', 'xx', 'xx'],
            edgecolor='black',
            label='Verification-only',
        )

    # Adding Xticks
    plt.xlabel('Random seed', fontweight='bold', fontsize=24)
    plt.ylabel('Runtime in seconds', fontweight='bold', fontsize=24)
    plt.xticks([r + barWidth for r in range(num_exp)], [str(i) for i in range(num_exp)])

    plt.yscale('log')
    plt.yticks(fontsize=22)
    plt.xticks(fontsize=22)
    plt.ylim([10, 5 * 1e4])

    legend_handles = [
        Patch(facecolor=accpm_bars[0].get_facecolor()),
        Patch(facecolor=verify_bars[0].get_facecolor()),
    ]

    # Add legend to the plot
    plt.legend(legend_handles, ['Fine-tuning', 'Verification-only'], fontsize=20)

    # Display the plot
    plt.show()
