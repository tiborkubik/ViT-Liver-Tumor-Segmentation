import os

modes = ['2D', '2.5D']
models = ['UNet', 'AttentionUNet', 'TransUNet']
losses = ['MSE', 'BCE']
weights_tumor = ["1.0", "2.0"]
options = [('-tm', modes), ('-n', models), ('-lo', losses), ('-wt', weights_tumor)]

non_acceptable_options = [('2.5D', 'TransUNet')]
edit_options = []
configs_path = 'configs'


def is_acceptable(config):
    is_ok = True
    for val in non_acceptable_options:
        all_options = all(True if option in config else False for option in val)
        if all_options:
            is_ok = False
            break
    return is_ok


def edit(config):
    for val in edit_options:
        if val[0] in config:
            config = config.replace(val[0], val[1])
    return config


def get_configs():
    all_configs = ['']
    for key, values in options:
        prev_configs = all_configs.copy()
        all_configs = [f"{config} {key if key else ''} {value}" for value in values for config in
                       prev_configs]
    all_configs = map(edit, filter(is_acceptable, all_configs))
    return list(all_configs)


if __name__ == "__main__":
    if os.path.exists(configs_path):
        for file in os.listdir(configs_path):
            os.remove(os.path.join(configs_path, file))
    else:
        os.makedirs(configs_path)

    all_configs = get_configs()
    # all_configs = ['-tm 2.5D -n UNet -lo MSE -wt 2.0 -b 16', '-tm 2.5D -n AttentionUNet -lo MSE -wt 2.0 -b 16', '-tm 2D -n TransUNet -lo MSE -wt 2.0 -b 4']
    for index, config in enumerate(all_configs):
        with open(f'configs/config{index + 1}.txt', 'w') as f:
            f.write(config)
