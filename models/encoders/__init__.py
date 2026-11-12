from .cftfm_8_6 import GET


def get_encoder(config, device):
   
    if config.name == 'GET':
        return GET(
            hidden_channels = config.hidden_channels,
            edge_channels = config.edge_channels,
            key_channels = config.key_channels,
            num_heads = config.num_heads,
            num_interactions = config.num_interactions,
            k = config.knn,
            cutoff = config.cutoff,
            device = device
        )
    else:
        raise NotImplementedError('Unknown encoder: %s' % config.name)