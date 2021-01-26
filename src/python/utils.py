'''
Useful helper functions
'''

import os
from os.path import join as fullfile
import platform
import numpy as np
import cv2 as cv
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms
import pytorch_ssim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import yaml

# for visualization
import visdom

vis = visdom.Visdom(port=8098, use_incoming_socket=False)  # default is 8097
assert vis.check_connection(), 'Visdom: No connection, start visdom first!'

# use qt5agg backend for remote interactive interpreter plot below
import matplotlib as mpl

# backend
mpl.use('Qt5Agg')
# mpl.use('TkAgg')

# disable toolbar and set background to black for full screen
mpl.rcParams['toolbar'] = 'None'
mpl.rcParams['figure.facecolor'] = 'black'

import matplotlib.pyplot as plt  # restart X11 session if it hangs (MobaXterm in my case)


# Use Pytorch multi-threaded dataloader and opencv to load image faster
class SimpleDataset(Dataset):
    """Simple dataset."""

    def __init__(self, data_root, index=None, size=None):
        self.data_root = data_root
        self.size = size

        # img list
        img_list = sorted(os.listdir(data_root))
        if index is not None: img_list = [img_list[x] for x in index]

        self.img_names = [fullfile(self.data_root, name) for name in img_list]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        assert os.path.isfile(img_name), img_name + ' does not exist'
        im = cv.imread(self.img_names[idx])

        # resize image if size is specified
        if self.size is not None:
            im = cv.resize(im, self.size[::-1])
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        return im


# set random number generators' seeds
def resetRNGseed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# read a single image to float tensor CxHxW
def readImg(file_name):
    return torch.Tensor(cv.cvtColor(cv.imread(file_name), cv.COLOR_BGR2RGB).transpose((2, 0, 1))) / 255


# read images using multi-thread
def readImgsMT(img_dir, size=None, index=None, gray_scale=False, normalize=False):
    img_dataset = SimpleDataset(img_dir, index=index, size=size)
    data_loader = DataLoader(img_dataset, batch_size=len(img_dataset), shuffle=False, drop_last=False, num_workers=4)

    for i, imgs in enumerate(data_loader):
        # imgs.permute((0, 3, 1, 2)).to('cpu', dtype=torch.float32)/255
        # convert to torch.Tensor
        imgs = imgs.permute((0, 3, 1, 2)).float().div(255)

        if gray_scale:
            imgs = 0.2989 * imgs[:, 0] + 0.5870 * imgs[:, 1] + 0.1140 * imgs[:, 2]  # same as MATLAB rgb2gray and OpenCV cvtColor
            imgs = imgs[:, None]

        # normalize to [-1, 1], may improve model convergence in early training stages.
        if normalize:
            imgs = (imgs - 0.5) / 0.5

        return imgs


# figure and show different type of tensors or ndarrays
def fs(inputData, title=None, cmap='gray'):
    inputData = inputData.squeeze()
    if type(inputData) is np.ndarray:
        im = inputData
    elif type(inputData) is torch.Tensor:
        F_tensor_to_image = torchvision.transforms.ToPILImage()

        if inputData.requires_grad:
            inputData = inputData.detach()

        if inputData.device.type == 'cuda':
            if inputData.ndimension() == 2:
                im = inputData.squeeze().cpu().numpy()
            else:
                im = F_tensor_to_image(inputData.squeeze().cpu())
        else:
            if inputData.ndimension() == 2:
                im = inputData.numpy()
            else:
                im = F_tensor_to_image(inputData.squeeze())

    # remove white paddings
    fig = plt.figure()
    # fig.canvas.window().statusBar().setVisible(False)

    # display image
    ax = plt.imshow(im, interpolation='bilinear', cmap=cmap, )
    ax = plt.axis('off')
    plt.tight_layout(pad=0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if title is not None:
        plt.title(title)
        plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0, hspace=0, wspace=0)
    plt.show()
    return fig


# save 4D np.ndarray or torch tensor to image files
def saveImgs(inputData, dir, idx=0):
    if not os.path.exists(dir):
        os.makedirs(dir)

    if type(inputData) is torch.Tensor:
        if inputData.requires_grad:
            inputData = inputData.detach()
        if inputData.device.type == 'cuda':
            imgs = inputData.cpu().numpy().transpose(0, 2, 3, 1)  # (N, C, row, col) to (N, row, col, C)
        else:
            imgs = inputData.numpy().transpose(0, 2, 3, 1)  # (N, C, row, col) to (N, row, col, C)

    else:
        imgs = inputData

    # imgs must have a shape of (N, row, col, C)
    if imgs.dtype == 'float32':
        imgs = np.uint8(imgs[:, :, :, ::-1] * 255)  # convert to BGR and uint8 for opencv
    else:
        imgs = imgs[:, :, :, ::-1]  # convert to BGR and uint8 for opencv
    for i in range(imgs.shape[0]):
        file_name = 'img_{:04d}.png'.format(i + 1 + idx)
        cv.imwrite(fullfile(dir, file_name), imgs[i, :, :, :])  # faster than PIL or scipy


# save estimated depth image to txt format, other formats (e.g., png, exr) may have bit issues or rounding errors
def saveDepth(depth, file_name):
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if type(depth) is not np.ndarray:
        depth = depth.numpy()

    # save as txt to avoid hdr or other image precision issue, downside is 10x file size increase.
    np.savetxt(file_name, depth, fmt='%1.4e')
    # cv.imwrite(file_name, depth)  # faster than PIL or scipy


# read inverse depth
def readDepth(file_name):
    # depth = 1 / cv.imread(file_name, cv.IMREAD_ANYDEPTH)[..., 0]
    # depth[depth == float('inf')] = 0
    depth = np.loadtxt(file_name)
    return depth


# check if each dataset has valid images and parameters
def checkDataList(dataset_root, data_list):
    for data_name in data_list:
        data_full_path = fullfile(dataset_root, data_name)
        assert os.path.exists(data_full_path), data_full_path + ' does not exist\n'

        param_file = fullfile(data_full_path, 'params/params.yml')
        assert os.path.exists(param_file), param_file + ' does not exist\n'


# convert yaml file string to np array
def stringToMat(m):
    n_rows = len(m)
    n_cols = len(m[0].split(','))

    mat = np.zeros((n_rows, n_cols))

    for r in range(n_rows):
        cur_row = m[r].split(',')
        for c in range(n_cols):
            mat[r][c] = float(cur_row[c])

    return mat


# load yml to torch tensor (Bx4x4)
def loadCalib(file_name):
    with open(file_name) as f:
        raw_data = yaml.load(f, yaml.Loader)

    calib_data = {}
    for m in raw_data:
        calib_data[m] = torch.Tensor(stringToMat(raw_data[m]))

    # calib_data['camRT'][0:3,0:3] = torch.eye(3,3)

    # convert to Kornia Bx4x4
    tensor_4x4 = torch.eye(4, 4)
    tensor_4x4[0:3, 0:3] = calib_data['camK']
    calib_data['camK'] = tensor_4x4.unsqueeze(0).clone()
    tensor_4x4[0:3, 0:3] = calib_data['prjK']
    calib_data['prjK'] = tensor_4x4.unsqueeze(0).clone()

    # extrinsics 3x4 ->1x4x4
    tensor_4x4[0:3, ...] = calib_data['camRT']
    calib_data['camRT'] = tensor_4x4.unsqueeze(0).clone()
    tensor_4x4[0:3, ...] = calib_data['prjRT']
    calib_data['prjRT'] = tensor_4x4.unsqueeze(0).clone()

    return calib_data


# compute PSNR
def psnr(x, y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device) if x.device.type != device.type else x
    y = y.to(device) if y.device.type != device.type else y

    with torch.no_grad():
        l2_fun = nn.MSELoss()
        return 10 * math.log10(1 / l2_fun(x, y))


# compute RMSE
def rmse(x, y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device) if x.device.type != device.type else x
    y = y.to(device) if y.device.type != device.type else y

    with torch.no_grad():
        l2_fun = nn.MSELoss()
        return math.sqrt(l2_fun(x, y).item() * 3)  # only works for RGB, for grayscale, don't multiply by 3


# compute SSIM
def ssim(x, y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device) if x.device.type != device.type else x
    y = y.to(device) if y.device.type != device.type else y

    with torch.no_grad():
        return pytorch_ssim.ssim(x, y).item()


# a differentiable function for true condition (TODO: change to torch.heaviside, which was just introduced in torch 1.7)
def softTrue(cond):
    scale = 1e4
    return torch.clamp(F.relu(cond) * scale, max=1)


# debug, using visdom to visualize images
def vfs(x, padding=10, title=None, ncol=None):
    nrow = 5 if ncol is None else ncol
    t = title if title is not None else ''

    if x.ndim == 3:
        vis.image(x, opts=dict(title=t, caption=t))
    elif x.ndim == 4 and x.shape[0] == 1:
        vis.image(x[0], opts=dict(title=t, caption=t))
    else:
        vis.images(x, opts=dict(title=t, caption=t), nrow=nrow, padding=padding)


# debug, visdom heatmap
def vhm(x, title=None):
    t = title if title is not None else ''
    vis.heatmap(x.squeeze(), opts=dict(title=t, caption=t, layoutopts=dict(plotly=dict(yaxis={'autorange': 'reversed'}))))


# training settings to string
def optionToString(train_option):
    return '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(train_option['data_name'], train_option['model_name'], train_option['loss'],
                                                  train_option['num_train'], train_option['batch_size'], train_option['max_iters'],
                                                  train_option['lr'], train_option['lr_drop_ratio'], train_option['lr_drop_rate'],
                                                  train_option['l2_reg'])


# print system config
def printConfig():
    print('-------------------------------------- System info -----------------------------------')

    # system
    print('OS: ', platform.platform())  # system build

    # pytorch
    print("torch version=" + torch.__version__)  # PyTorch version
    print("CUDA version=" + torch.version.cuda)  # Corresponding CUDA version
    # print("CUDNN version=" + torch.backends.cudnn.version())  # Corresponding cuDNN version

    # check GPU
    if torch.cuda.device_count() >= 1:
        print('Train with', torch.cuda.device_count(), 'GPUs!')
    else:
        print('Train with CPU!')

    # GPU name
    for i in range(torch.cuda.device_count()):
        print("GPU {:d} name: ".format(i) + torch.cuda.get_device_name(i))  # GPU name

    print('-------------------------------------- System info -----------------------------------')


