import hashlib
import os


def generate_bash(cmds, output_bash_path, per_node_commands):
    # bash_txt = '''export PYTHONPATH=$PYTHONPATH:/Users/shaoruchen/Documents/GitHub/learning-basis-functions/codes/learning_barrier \n'''
    bash_txt = ''
    for start_idx in range(0, len(cmds), per_node_commands):
        cmd_batch = cmds[start_idx : start_idx + per_node_commands]
        # add python command
        wait_suffix = ' & wait' if len(cmd_batch) > 1 else ''

        python_cmd = ' & '.join(_['cmd'] for _ in cmd_batch)

        bash_txt += f'{python_cmd} {wait_suffix} '

        bash_txt += '\n'

    # write yaml file
    with open(output_bash_path, 'w') as bash_file:
        bash_file.write(bash_txt)


def main():
    # generate python commands for grid-search

    cmds = []

    for seed in range(8):
        # for method in ['fine-tuning', 'verification-only']:
        for method in ['fine-tuning']:
            # for method in ['verification-only']:

            cmd = (
                f'python main_iter_train_double_integrator.py '
                f' --train_method {method}'
                f' --seed {seed}'
                f' --config double_integrator.yaml'
            )

            cmds.append({'job_name': hashlib.md5(cmd.encode()).hexdigest(), 'cmd': cmd})

    # submit left out cmds to generic cluster

    parallel_commands = 1

    if len(cmds) > 0:
        generate_bash(
            cmds,
            os.path.join(os.getcwd(), 'commands.sh'),
            per_node_commands=parallel_commands,
        )


if __name__ == '__main__':
    main()
