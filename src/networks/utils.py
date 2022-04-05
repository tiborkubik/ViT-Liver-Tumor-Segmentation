import torch

from src.networks.AttentionUNet import AttentionUNet
from src.networks.UNet import UNet


def create_model(network_name: str, weights_path: str = None):
    in_channels = 1  # BW images
    out_channels = 2  # Liver & Tumor mask

    assert network_name in ['UNet', 'AttentionUNet', 'TransUNet']

    if network_name == 'UNet':
        network = UNet(in_channels=in_channels, out_channels=out_channels,
                       batch_norm=True, decoder_mode='upconv')
    elif network_name == 'AttentionUNet':
        network = AttentionUNet(in_channels=in_channels, out_channels=out_channels)
    else:
        network = NestedUNet(in_channels=in_channels, out_channels=out_channels)

    if weights_path is not None:
        if torch.cuda.is_available():
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = 'cpu'

        weights = torch.load(weights_path, map_location=map_location)
        network.load_state_dict(weights)

    return network
