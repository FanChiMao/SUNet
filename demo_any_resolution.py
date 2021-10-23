import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import os
import utils
from skimage import img_as_ubyte
from collections import OrderedDict
from natsort import natsorted
from glob import glob
import cv2
import argparse
from model.SUNet import SUNet_model
import math
import tqdm
import yaml

with open('training.yaml', 'r') as config:
    opt = yaml.safe_load(config)


parser = argparse.ArgumentParser(description='Demo Image Restoration')
parser.add_argument('--input_dir', default='D:/pythonProject/SUNet/', type=str, help='Input images')
parser.add_argument('--window_size', default=8, type=int, help='window size (fixed with training)')
parser.add_argument('--result_dir', default='./result/', type=str, help='Directory for results')
parser.add_argument('--weights',
                    default='./pretrained-model/model_bestPSNR.pth', type=str,
                    help='Path to weights')

args = parser.parse_args()

def expand2square(timg, factor=16.0):
    _, _, h, w = timg.size()
    # 321, 481
    X = int(math.ceil(max(h, w) / float(factor)) * factor)
    img = torch.zeros(1, 3, X, X).type_as(timg)  # 3, h, w
    mask = torch.zeros(1, 1, X, X).type_as(timg)
    img[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)] = timg
    mask[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)].fill_(1.0)

    return img, mask


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

inp_dir = args.input_dir
out_dir = args.result_dir

os.makedirs(out_dir, exist_ok=True)

files = natsorted(glob(os.path.join(inp_dir, '*.jpg'))
                  + glob(os.path.join(inp_dir, '*.JPG'))
                  + glob(os.path.join(inp_dir, '*.png'))
                  + glob(os.path.join(inp_dir, '*.PNG')))

if len(files) == 0:
    raise Exception(f"No files found at {inp_dir}")

# Load corresponding model architecture and weights
model = SUNet_model(opt)
model.cuda()

load_checkpoint(model, args.weights)
model.eval()

print('restoring images......')
for file_ in files:
    img = Image.open(file_).convert('RGB')
    input_ = TF.to_tensor(img).unsqueeze(0).cuda()

    with torch.no_grad():
        square_input_, mask = expand2square(input_.cuda(), factor=256)
        restored = model(square_input_)
        restored = torch.masked_select(restored, mask.bool()).reshape(1, 3, input_.shape[2], input_.shape[3])
        restored = torch.clamp(restored, 0, 1)
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

    restored = img_as_ubyte(restored[0])

    f = os.path.splitext(os.path.split(file_)[-1])[0]
    save_img((os.path.join(out_dir, f + '.png')), restored)

print(f"Files saved at {out_dir}")
print('finish !')