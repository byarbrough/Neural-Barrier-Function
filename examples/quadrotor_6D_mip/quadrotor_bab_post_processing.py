import sys
# Append the path to the directory containing your module to sys.path
import os
project_dir = '/home/shaoruchen/Desktop/learning_basis/learning-basis-functions/codes/learning_barrier'
sys.path.append(project_dir)

script_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_directory)

result_dir = '/home/shaoruchen/Desktop/learning_basis/learning-basis-functions/codes/learning_barrier/examples/quadrotor_6D/data_0926'
import torch

num_exp = 8
accpm_result= {'status': [], 'timeout_status':[], 'train_time':[], 'verification_time':[], 'runtime':[]}
for i in range(num_exp):
    accpm_result_path = os.path.join(result_dir, 'train_iter_accpm_'+str(i)+'.p')
    result = torch.load(accpm_result_path)
    accpm_result['status'].append(result['training_results']['status'])
    accpm_result['timeout_status'].append(result['training_results']['timeout_status'])
    accpm_result['train_time'].append(sum(result['training_results']['train_time']))
    accpm_result['verification_time'].append(sum(result['training_results']['verification_time']))
    accpm_result['runtime'].append(sum(result['training_results']['train_time']) + sum(result['training_results']['verification_time']))

verify_result = {'status': [], 'timeout_status': [], 'train_time': [], 'verification_time': [], 'runtime': []}
for i in range(num_exp):
    verify_result_path = os.path.join(result_dir, 'train_iter_verify_' + str(i) + '.p')
    result = torch.load(verify_result_path)
    verify_result['status'].append(result['training_results']['status'])
    verify_result['timeout_status'].append(result['training_results']['timeout_status'])
    verify_result['train_time'].append(sum(result['training_results']['train_time']))
    verify_result['verification_time'].append(sum(result['training_results']['verification_time']))
    verify_result['runtime'].append(sum(result['training_results']['train_time'])+sum(result['training_results']['verification_time']))

print('')

