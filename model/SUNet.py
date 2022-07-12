import torch.nn as nn
from model.SUNet_detail import SUNet


class SUNet_model(nn.Module):
    def __init__(self, config):
        super(SUNet_model, self).__init__()
        self.config = config
        self.swin_unet = SUNet(img_size=config['SWINUNET']['IMG_SIZE'],
                               patch_size=config['SWINUNET']['PATCH_SIZE'],
                               in_chans=3,
                               out_chans=3,
                               embed_dim=config['SWINUNET']['EMB_DIM'],
                               depths=config['SWINUNET']['DEPTH_EN'],
                               num_heads=config['SWINUNET']['HEAD_NUM'],
                               window_size=config['SWINUNET']['WIN_SIZE'],
                               mlp_ratio=config['SWINUNET']['MLP_RATIO'],
                               qkv_bias=config['SWINUNET']['QKV_BIAS'],
                               qk_scale=config['SWINUNET']['QK_SCALE'],
                               drop_rate=config['SWINUNET']['DROP_RATE'],
                               drop_path_rate=config['SWINUNET']['DROP_PATH_RATE'],
                               ape=config['SWINUNET']['APE'],
                               patch_norm=config['SWINUNET']['PATCH_NORM'],
                               use_checkpoint=config['SWINUNET']['USE_CHECKPOINTS'])

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.swin_unet(x)
        return logits
    
if __name__ == '__main__':
    from utils.model_utils import network_parameters
    import torch
    import yaml
    from thop import profile
    from utils.model_utils import network_parameters

    ## Load yaml configuration file
    with open('../training.yaml', 'r') as config:
        opt = yaml.safe_load(config)
    Train = opt['TRAINING']
    OPT = opt['OPTIM']

    height = 256
    width = 256
    x = torch.randn((1, 156, height, width))  # .cuda()
    model = SUNet_model(opt)  # .cuda()
    out = model(x)
    flops, params = profile(model, (x,))
    print(out.size())
    print(flops)
    print(params)
