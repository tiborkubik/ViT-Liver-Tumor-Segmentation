import os
import matplotlib.pyplot as plt
import numpy as np


def rename_experiment(experiment):
    if 'liver' in experiment:
        return f'lk:{experiment.split(":")[1].strip()}'
    if 'tumor' in experiment:
        return f'tk:{experiment.split(":")[1].strip()}'
    if 'masking' in experiment:
        if 'True' in experiment:
            return 'mask'
        else:
            return ''
    if 'morph' in experiment:
        if 'True' in experiment:
            return 'morph'
        else:
            return ''


def parse_experiments(experiments):
    runs = experiments.split('\n')[:-1]
    parsed_e = []
    for i in range(len(runs) // 6):
        setting = runs[i * 6]
        settings = ' '.join(setting.split(' ')[1:]).split(',')
        experiment = {
            'setting': {f'{setting.split(":")[0].strip()}': setting.split(":")[1].strip() for setting in settings},
            'type': setting.split(' ')[0]}
        for metric in range(1, 6):
            metric_name, value = runs[i * 6 + metric].split(':')
            experiment[metric_name] = .0 if value.strip() == 'None' else float(value)
        parsed_e.append(experiment)
    return parsed_e


def get_kernel_stats(run, kernels, type):
    per_kernel_dice = {setting['setting']['liver_kernel']: setting['Dice'] for setting in run if
                       setting['setting']['liver_kernel'] in kernels and setting['setting'][
                           'apply_masking'] == 'False' and setting['type'] == type}
    return np.array([per_kernel_dice[kernel] for kernel in kernels])


def get_masking_stats(run):
    masking_true = [setting['Dice'] for setting in run if setting['setting'][
        'apply_masking'] == 'True' and setting['setting'][
                        'apply_morphological'] == 'False' and setting['type'] == 'Lesion']
    masking_false = [setting['Dice'] for setting in run if setting['setting'][
        'apply_masking'] == 'False' and setting['setting'][
                         'apply_morphological'] == 'False' and setting['type'] == 'Lesion']
    return np.array([masking_false[0], masking_true[0]])


def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]


def print_result(dice_scores, configs, name,condition):
    print(f"{name}: {np.max(dice_scores[find_indices(configs, condition), :])}")


def print_best_results(dice_scores, configs):
    print_result(dice_scores, configs, 'UNet 2D', lambda x: '2D' in x and ' UNet' in x)
    print_result(dice_scores, configs, 'UNet 2.5D', lambda x: '2.5D' in x and ' UNet' in x)
    print_result(dice_scores, configs, 'AttentionUNet 2D', lambda x: '2D' in x and 'AttentionUNet' in x)
    print_result(dice_scores, configs, 'AttentionUNet 2.5D', lambda x: '2.5D' in x and 'AttentionUNet' in x)
    print_result(dice_scores, configs, 'TransUNet 2D', lambda x: '2D' in x and 'TransUNet' in x)


def visualize_results(runs, configs):
    liver_kernels = ['0', '2', '4', '16']
    lesion_kernels = ['0', '2', '4']
    dice_scores_liver = []
    dice_scores_tumor = []
    for run in runs:
        dice_scores_liver.append(get_kernel_stats(run, liver_kernels, 'Liver'))
        dice_scores_tumor.append(get_kernel_stats(run, lesion_kernels, 'Lesion'))
    dice_scores_liver = np.stack(dice_scores_liver)
    dice_scores_tumor = np.stack(dice_scores_tumor)
    liver_detected = np.all(dice_scores_liver > 0, axis=1)
    tumor_detected = np.all(dice_scores_tumor > 0, axis=1)
    dice_scores_liver = dice_scores_liver[liver_detected, :]
    dice_scores_tumor = dice_scores_tumor[tumor_detected, :]

    fig, axs = plt.subplots(1, 2, figsize=(7, 4))
    # fig.suptitle('Comparison of different kernels used for morphological transformations')
    axs[0].boxplot(dice_scores_liver)
    axs[0].set_xlabel('Kernel size')
    axs[0].set_ylabel('Dice score')
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].set_title('Liver')
    axs[0].set_xticks(range(1, 5), liver_kernels)

    axs[1].boxplot(dice_scores_tumor)
    axs[1].set_xlabel('Kernel size')
    axs[1].set_ylabel('Dice score')
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].set_title('Lesion')
    axs[1].set_xticks(range(1, 4), lesion_kernels)

    fig.savefig('documentation/morphological_kernel.pdf')

    masking = np.stack([get_masking_stats(run) for run in runs])
    masking = masking[np.all(masking > 0, axis=1), :]

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.boxplot(masking)
    ax.set_xlabel('Masking')
    ax.set_ylabel('Dice score')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # axs[0].set_title('Comparison of ')
    ax.set_xticks(range(1, 3), ['Disabled', "Enabled"])
    fig.savefig('documentation/masking.pdf')

    print('Liver')
    print_best_results(dice_scores_liver, [config for index, config in enumerate(configs) if liver_detected[index]])
    print('Tumor')
    print_best_results(dice_scores_tumor, [config for index, config in enumerate(configs) if tumor_detected[index]])


if __name__ == '__main__':
    configs = os.listdir('configs')
    results = []
    configs_loaded = []
    for index, config in enumerate(configs):
        with open(os.path.join('configs', config)) as f:
            configs_loaded.append(f.read())
            with open(os.path.join('experiments', str(index + 1), 'metrics.log')) as fr:
                results.append(parse_experiments(fr.read()))
    visualize_results(results, configs_loaded)
    x = 2
