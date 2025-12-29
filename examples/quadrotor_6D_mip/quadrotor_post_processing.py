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
    result_dir = os.path.join(script_directory, 'data_0209_mip')
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

    # bab results
    accpm_bab_result = {}
    accpm_bab_result['status'] = [
        'feasible',
        'feasible',
        'feasible',
        'feasible',
        'feasible',
        'feasible',
        'feasible',
        'feasible',
    ]
    accpm_bab_result['runtime'] = [
        119.78482308836634,
        245.47588879739394,
        339.50341095674685,
        54.03175181743506,
        259.12558347837194,
        326.3891123580156,
        82.29843147572412,
        279.47603725561305,
    ]
    accpm_bab_result['train_time'] = [
        44.70002555847168,
        27.77279233932495,
        32.47540545463562,
        23.96451187133789,
        36.568394899368286,
        12.129829406738281,
        67.97045683860779,
        36.23796224594116,
    ]
    accpm_bab_result['verification_time'] = [
        75.08479752989466,
        217.70309645806898,
        307.0280055021112,
        30.067239946097168,
        222.55718857900365,
        314.2592829512773,
        14.327974637116332,
        243.2380750096719,
    ]
    accpm_bab_result['timeout_status'] = [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]

    verify_bab_result = {}

    verify_bab_result['status'] = [
        'feasible',
        'unknown',
        'feasible',
        'unknown',
        'feasible',
        'feasible',
        'feasible',
        'feasible',
    ]
    verify_bab_result['runtime'] = [
        2671.555299282074,
        7491.943889379501,
        1614.4262449741364,
        7427.887099266052,
        6860.558847904205,
        1398.2325344085693,
        283.56421279907227,
        2989.3616967201233,
    ]
    verify_bab_result['train_time'] = [
        406.4860050678253,
        791.3171737194061,
        338.56566739082336,
        790.3344197273254,
        750.5411593914032,
        229.63532423973083,
        141.31337237358093,
        438.24117398262024,
    ]
    verify_bab_result['verification_time'] = [
        2265.0692942142487,
        6700.626715660095,
        1275.860577583313,
        6637.552679538727,
        6110.017688512802,
        1168.5972101688385,
        142.25084042549133,
        2551.120522737503,
    ]
    verify_bab_result['timeout_status'] = [
        False,
        True,
        False,
        True,
        False,
        False,
        False,
        False,
    ]

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
            hatch=['', 'xx', '', '', '', '', 'xx', ''],
            edgecolor='black',
            label='Verification-only',
        )
    else:
        accpm_bars = plt.bar(
            br1,
            accpm_bab_result['runtime'],
            width=barWidth,
            hatch=['', '', '', '', '', '', '', ''],
            edgecolor='black',
            label='Fine-tuning',
        )

        verify_bars = plt.bar(
            br2,
            verify_bab_result['runtime'],
            width=barWidth,
            hatch=['', 'xx', '', 'xx', '', '', '', ''],
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

    plt.show()
