# Copyright (C) 2020 Michael Mommert
# This file is part of IndustrialSmokePlumeDetection
# <https://github.com/HSG-AIML/IndustrialSmokePlumeDetection>.
#
# IndustrialSmokePlumeDetection is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# IndustrialSmokePlumeDetection is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with IndustrialSmokePlumeDetection.  If not,
# see <http://www.gnu.org/licenses/>.
#
# If you use this code for your own project, please cite the following
# conference contribution:
#   Mommert, M., Sigel, M., Neuhausler, M., Scheibenreif, L., Borth, D.,
#   "Characterization of Industrial Smoke Plumes from Remote Sensing Data",
#   Tackling Climate Change with Machine Learning workshop at NeurIPS 2020.
#
#
# This file contains a wrapper for the training of the segmentation model.

import numpy as np
import torch
from torch import nn, optim
from tqdm.autonotebook import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, random_split, RandomSampler
from torch.utils.tensorboard import SummaryWriter
import argparse
from sklearn.metrics import jaccard_score

from model_unet import *
from data import create_dataset

print('running on...', device)

def train_model(model, epochs, opt, loss, batch_size):
    """Wrapper function for model training.

    :param model: model instance
    :param epochs: (int) number of epochs to be trained
    :param opt: optimizer instance
    :param loss: loss function instance
    :param batch_size: (int) batch size"""

    # create dataset
    data_train = create_dataset(
        datadir='/path/to/image/data/',
        seglabeldir='/path/to/segmentation/labels/for/training/', mult=1)
    data_val = create_dataset(
        datadir='/path/to/image/data/',
        seglabeldir='/path/to/segmentation/labels/for/validation/', mult=1)

    # draw random subsamples
    train_sampler = RandomSampler(data_train, replacement=True,
                                  num_samples=int(2*len(data_train)/3))
    val_sampler = RandomSampler(data_val, replacement=True,
                                 num_samples=int(2*len(data_val)/3))

    # initialize data loaders
    train_dl = DataLoader(data_train, batch_size=batch_size, num_workers=6,
                          pin_memory=True, sampler=train_sampler)
    val_dl = DataLoader(data_val, batch_size=batch_size, num_workers=6,
                         pin_memory=True, sampler=val_sampler)

    # start training process
    for epoch in range(epochs):

        model.train()

        train_loss_total = 0
        train_ious = []
        train_acc_total = 0
        train_arearatios = []
        progress = tqdm(enumerate(train_dl), desc="Train Loss: ",
                        total=len(train_dl))
        for i, batch in progress:
            x = batch['img'].float().to(device)
            y = batch['fpt'].float().to(device)

            output = model(x)

            # derive binary segmentation map from prediction
            output_binary = np.zeros(output.shape)
            output_binary[output.cpu().detach().numpy() >= 0] = 1

            # derive IoU values
            ious = []
            for j in range(y.shape[0]):
                z = jaccard_score(y[j].flatten().cpu().detach().numpy(),
                          output_binary[j][0].flatten())
                if (np.sum(output_binary[j][0]) != 0 and
                    np.sum(y[j].cpu().detach().numpy()) != 0):
                    train_ious.append(z)

            # derive scalar binary labels on a per-image basis
            y_bin = np.array(np.sum(y.cpu().detach().numpy(),
                                    axis=(1,2)) != 0).astype(int)
            pred_bin = np.array(np.sum(output_binary,
                                       axis=(1,2,3)) != 0).astype(int)

            # derive image-wise accuracy for this batch
            train_acc_total += accuracy_score(y_bin, pred_bin)

            # derive loss
            loss_epoch = loss(output, y.unsqueeze(dim=1))
            train_loss_total += loss_epoch.item()
            progress.set_description("Train Loss: {:.4f}".format(
                train_loss_total/(i+1)))

            # derive smoke areas
            area_pred = np.sum(output_binary, axis=(1,2,3))
            area_true = np.sum(y.cpu().detach().numpy(), axis=(1,2))

            # derive smoke area ratios
            arearatios = []
            for k in range(len(area_pred)):
                if area_pred[k] == 0 and area_true[k] == 0:
                    arearatios.append(1)
                elif area_true[k] == 0:
                    arearatios.append(0)
                else:
                    arearatios.append(area_pred[k]/area_true[k])
            train_arearatios = np.ravel([*train_arearatios, *arearatios])

            # learning
            opt.zero_grad()
            loss_epoch.backward()
            opt.step()

        # logging
        writer.add_scalar("training loss", train_loss_total/(i+1), epoch)
        writer.add_scalar("training iou", np.average(train_ious), epoch)
        writer.add_scalar("training acc", train_acc_total/(i+1), epoch)
        writer.add_scalar('training arearatio mean',
                          np.average(train_arearatios), epoch)
        writer.add_scalar('training arearatio std',
                          np.std(train_arearatios), epoch)
        writer.add_scalar('learning_rate', opt.param_groups[0]['lr'], epoch)

        torch.cuda.empty_cache()

        # evaluation
        model.eval()
        val_loss_total = 0
        val_ious = []
        val_acc_total = 0
        val_arearatios = []
        progress = tqdm(enumerate(val_dl), desc="val Loss: ",
                        total=len(val_dl))
        for j, batch in progress:
            x = batch['img'].float().to(device)
            y = batch['fpt'].float().to(device)

            output = model(x)

            # derive loss
            loss_epoch = loss(output, y.unsqueeze(dim=1))
            val_loss_total += loss_epoch.item()

            # derive binary segmentation map from prediction
            output_binary = np.zeros(output.shape)
            output_binary[output.cpu().detach().numpy() >= 0] = 1

            # derive IoU values
            ious = []
            for k in range(y.shape[0]):
                z = jaccard_score(y[k].flatten().cpu().detach().numpy(),
                          output_binary[k][0].flatten())
                if (np.sum(output_binary[k][0]) != 0 and 
                    np.sum(y[k].cpu().detach().numpy()) != 0):
                    val_ious.append(z)

            # derive scalar binary labels on a per-image basis
            y_bin = np.array(np.sum(y.cpu().detach().numpy(),
                                    axis=(1,2)) != 0).astype(int)
            pred_bin = np.array(np.sum(output_binary,
                                       axis=(1,2,3)) != 0).astype(int)

            # derive image-wise accuracy for this batch
            val_acc_total += accuracy_score(y_bin, pred_bin)

            # derive smoke areas
            area_pred = np.sum(output_binary, axis=(1,2,3))
            area_true = np.sum(y.cpu().detach().numpy(), axis=(1,2))

            # derive smoke area ratios
            arearatios = []
            for k in range(len(area_pred)):
                if area_pred[k] == 0 and area_true[k] == 0:
                    arearatios.append(1)
                elif area_true[k] == 0:
                    arearatios.append(0)
                else:
                    arearatios.append(area_pred[k]/area_true[k])
            val_arearatios = np.ravel([*val_arearatios, *arearatios])
            
            progress.set_description("val Loss: {:.4f}".format(
                val_loss_total/(j+1)))

        # logging
        writer.add_scalar("val loss", val_loss_total/(j+1), epoch)
        writer.add_scalar("val iou", np.average(val_ious), epoch)
        writer.add_scalar("val acc", val_acc_total/(j+1), epoch)
        writer.add_scalar('val arearatio mean',
                          np.average(val_arearatios), epoch)
        writer.add_scalar('val arearatio std',
                          np.std(val_arearatios), epoch)
        
        print(("Epoch {:d}: train loss={:.3f}, val loss={:.3f}, "
               "train iou={:.3f}, val iou={:.3f}, "
               "train acc={:.3f}, val acc={:.3f}").format(
                   epoch+1, train_loss_total/(i+1), val_loss_total/(j+1),
                   np.average(train_ious), np.average(val_ious),
                   train_acc_total/(i+1), val_acc_total/(j+1)))
      
        # save model checkpoint
        if epoch % 1 == 0:
            torch.save(model.state_dict(),
            'ep{:0d}_lr{:.0e}_bs{:02d}_mo{:.1f}_{:03d}.model'.format(
                args.ep, args.lr, args.bs, args.mo, epoch))

        writer.flush()
        scheduler.step(val_loss_total/(j+1))
        torch.cuda.empty_cache()

    return model


# setup argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-ep', type=int, default=300,
                    help='Number of epochs')
parser.add_argument('-bs', type=int, nargs='?',
                    default=60, help='Batch size')
parser.add_argument('-lr', type=float,
                    nargs='?', default=0.7, help='Learning rate')
parser.add_argument('-mo', type=float,
                    nargs='?', default=0.7, help='Momentum')
args = parser.parse_args()

# setup tensorboard writer
writer = SummaryWriter('runs/'+"ep{:0d}_lr{:.0e}_bs{:03d}_mo{:.1f}/".format(
    args.ep, args.lr, args.bs, args.mo))

# initialize loss function
loss = nn.BCEWithLogitsLoss()

# initialize optimizer
opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mo)

# initialize scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min',
                                                 factor=0.5, threshold=1e-4,
                                                 min_lr=1e-6)

# run training
train_model(model, args.ep, opt, loss, args.bs)

writer.close()

