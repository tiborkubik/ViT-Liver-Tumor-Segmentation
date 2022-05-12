import os

modes = ['2D', '2.5D']
models = ['UNet', 'AttentionUNet', 'TransUNet']
losses = ['MSE', 'BCE']
weights_liver = ["1.0", "2.0"]
options = [('-tm', modes), ('-n', models), ('-lo', losses), ('-wl', weights_liver)]

non_acceptable_options = [('2.5D', 'TransUNet')]
edit_options = [('2D', '2D -b 128'), ('-b 128 -n TransUNet', '-b 16 -n TransUNet')]
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


if __name__ == "__main__":
    if os.path.exists(configs_path):
        for file in os.listdir(configs_path):
            os.remove(os.path.join(configs_path, file))
    else:
        os.makedirs(configs_path)

    all_configs = ['']
    for key, values in options:
        prev_configs = all_configs.copy()
        all_configs = [f"{config} {key if key else ''} {value}" for value in values for config in
                       prev_configs]
    all_configs = map(edit, filter(is_acceptable, all_configs))
    for index, config in enumerate(all_configs):
        with open(f'configs/config{index + 1}.txt', 'w') as f:
            f.write(config)
