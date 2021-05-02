# imports
import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
from fastai.vision.all import \
        ranger, Metric, Path, ImageDataLoaders, SaveModelCallback, Learner, flatten_check
# from deepflash2.all import *
import albumentations as alb
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from torch import Tensor, IntTensor
from segmentation_models_pytorch.losses import JaccardLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fold", default=0, type=int,
                    help="fold")
args = parser.parse_args()

class Dice_soft(Metric):
    def __init__(self, axis=1):
        self.axis = axis

    def reset(self): self.inter, self.union = 0, 0

    def accumulate(self, learn):
        pred, targ = flatten_check(torch.sigmoid(learn.pred), learn.y)
        self.inter += (pred*targ).float().sum().item()
        self.union += (pred+targ).float().sum().item()

    @property
    def value(self): return 2.0 * self.inter / \
        self.union if self.union > 0 else None

# dice with automatic threshold selection


class Dice_th(Metric):
    def __init__(self, ths=np.arange(0.1, 0.9, 0.05), axis=1):
        self.axis = axis
        self.ths = ths

    def reset(self):
        self.inter = torch.zeros(len(self.ths))
        self.union = torch.zeros(len(self.ths))

    def accumulate(self, learn):
        pred, targ = flatten_check(torch.sigmoid(learn.pred), learn.y)
        for i, th in enumerate(self.ths):
            p = (pred > th).float()
            self.inter[i] += (p*targ).float().sum().item()
            self.union[i] += (p+targ).float().sum().item()

    @property
    def value(self):
        dices = torch.where(self.union > 0.0,
                            2.0*self.inter/self.union, torch.zeros_like(self.union))
        return dices.max()


class CONFIG():
    debug = False

    # data paths
    main_path = Path('../input/hubmap-kidney-segmentation')
    data_path = Path('../input/512x512-reduce-2/train')
    label_path = Path('../input/512x512-reduce-2/masks')
    csv_path = '../input/hubmap-enhanced-masks/all_masks_v4.csv'

    # use test data
    use_test = False

    # folds
    nfolds = 5

    # seed
    SEED = 2020

    # deepflash2 dataset
    scale = 1  # data is already downscaled to 2, so absulute downscale is 3
    tile_shape = (1024, 1024)
    padding = (0, 0)  # Border overlap for prediction
    n_jobs = 1
    sample_mult = 500  # Sample 100 tiles from each image, per epoch
    val_length = 500  # Randomly sample 500 validation tiles

    # TODO: Change these as per the dataset's mean and std
    mean = np.array([0.63808466, 0.47418504, 0.68202581])
    std = np.array([0.16178218, 0.22919784, 0.14178402])

    # deepflash2 augmentation options
    zoom_sigma = 0.1
    flip = True
    max_rotation = 360
    deformation_grid_size = (150, 150)
    deformation_magnitude = (10, 10)

    # pytorch model (segmentation_models_pytorch)
    encoder_name = "timm-efficientnet-b4"
    encoder_weights = 'advprop'
    in_channels = 3
    classes = 1

    # fastai Learner
    mixed_precision_training = True
    batch_size = 32
    weight_decay = 0.01
    loss_func = torch.nn.BCEWithLogitsLoss()
    metrics = [Dice_soft(), Dice_th()]
    optimizer = ranger
    max_learning_rate = 3e-3
    epochs = 16
    if debug:
        epochs=4
        batch_size=24


cfg = CONFIG()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # the following line gives ~10% speedup
    # but may lead to some stochasticity in the results
    torch.backends.cudnn.benchmark = True


seed_everything(cfg.SEED)

tfms = alb.Compose([
    alb.HorizontalFlip(),
    alb.VerticalFlip(),
    alb.RandomRotate90(),
    alb.RandomBrightness(),
    alb.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9,
                         border_mode=cv2.BORDER_REFLECT),
    alb.OneOf([
        alb.HueSaturationValue(10, 15, 10),
        alb.CLAHE(clip_limit=2),
        alb.RandomBrightnessContrast(),
    ], p=0.3),
], p=0.8)


df_train = pd.read_csv(cfg.csv_path)
if cfg.use_test:
    pass
else:
    print("Not using the test data for training the model...")
    df_train = df_train[df_train['set'] != 'test']
print(df_train)


# external_data = [x.replace('.png', '').replace("external_", "") for x in os.listdir(cfg.data_path) if '.png' in x if 'external' in x]
imgs_idxs = [x.replace('.png', '')
             for x in os.listdir(cfg.data_path) if '.png' in x]
if cfg.debug:
    imgs_idxs = imgs_idxs[:cfg.nfolds*cfg.batch_size*2]

for iname in list(set([x[:9] for x in imgs_idxs])):
    print('img name:', iname,
          '| imgs number:', len([x for x in imgs_idxs if x[:9] == iname]))
kfold = KFold(n_splits=cfg.nfolds,
              random_state=cfg.SEED,
              shuffle=True).split(imgs_idxs)


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


class HuBMAPDataset(Dataset):
    def __init__(self, fnames, train=True, tfms=None):
        self.fnames = fnames
        self.train = train
        self.tfms = tfms

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]

        img_path = f'{cfg.data_path}/{fname}.png'
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is None:
            img_path = f'{cfg.data_path}/external_{fname}.png'
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        msk_path = f'{cfg.label_path}/{fname}.png'
        mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            msk_path = f'{cfg.label_path}/external_{fname}.png'
            mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        if self.tfms is not None:
            augmented = self.tfms(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']
        return img2tensor((img/255.0 - cfg.mean)/cfg.std), img2tensor(mask)

name = 'Unet'
base_model = smp.Unet(encoder_name=cfg.encoder_name,
                              encoder_weights=cfg.encoder_weights,
                              in_channels=cfg.in_channels,
                              classes=cfg.classes)


class HuBMAPModel(nn.Module):
    def __init__(self):
        super(HuBMAPModel, self).__init__()
        self.cnn_model = base_model

    def forward(self, imgs):
        img_segs = self.cnn_model(imgs)
        return img_segs


for n, (tr, te) in enumerate(kfold):
    fold = n
    print('=' * 10, f'FOLD {n}', '=' * 10)
    X_tr = [imgs_idxs[i] for i in tr]
    X_val = [imgs_idxs[i] for i in te]
    if False:#cfg.debug:
        X_tr = X_tr[:32]
        X_val = X_val[:32]
    print('train:', len(X_tr), '| test:', len(X_val))
    print('groups train:', set([x[:9] for x in X_tr]),
            '\ngroups test:', set([x[:9] for x in X_val]))

    model = HuBMAPModel()

    train_ds = HuBMAPDataset(X_tr, tfms)
    valid_ds = HuBMAPDataset(X_val)

    data = ImageDataLoaders.from_dsets(train_ds, valid_ds, bs=cfg.batch_size,
                                        num_workers=4*4, pin_memory=True)

    if torch.cuda.is_available():
        data.cuda(), model.cuda()

    cbs = [SaveModelCallback(monitor='dice_th', comp=np.greater)]
    learn = Learner(data, model, metrics=cfg.metrics, wd=cfg.weight_decay,
                    loss_func=cfg.loss_func, opt_func=ranger, cbs=cbs, model_dir='models'+'_'+name)
    if cfg.mixed_precision_training:
        learn.to_fp16()

    # make learner to use all GPUs
    learn.model = torch.nn.DataParallel(learn.model)
    if fold != args.fold:
        continue

    # Fit
    learn.fit_one_cycle(cfg.epochs, lr_max=cfg.max_learning_rate)

    # Save Model
    state = {'model': learn.model.state_dict(), 'mean': cfg.mean,
                'std': cfg.std}
    torch.save(state, f'{name}_{cfg.encoder_name}_{fold}.pth',
                pickle_protocol=2, _use_new_zipfile_serialization=False)
