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
# This file contains the data handling infrastructure for the segmentation
# model.

import os
import numpy as np
import json
from matplotlib import pyplot as plt
import rasterio as rio
from rasterio.features import rasterize
from shapely.geometry import Polygon
import torch
from torchvision import transforms

# set random seeds
torch.manual_seed(3)
np.random.seed(3)

# data directory
outdir = os.path.abspath('.')

class SmokePlumeSegmentationDataset():
    """SmokePlumeSegmentation dataset class."""

    def __init__(self,
                 datadir=None, seglabeldir=None, mult=1,
                 transform=None):
        """SmokePlumeSegmentation Dataset class.

        The data set built will contain as many negative examples as there are
        positive examples to enfore balancing.

        The function `create_dataset` can be used as a wrapper to create a
        data set.

        :param datadir: (str) image directory root, required
        :param seglabeldir: (str) segmentation label directory root, required
        :param mult: (int) factor by which to multiply data set size, default=1
        :param transform: (`torchvision.transform` object) transformations to be
                          applied, default: `None`
        """
        
        self.datadir = datadir
        self.transform = transform

        # list of image files, labels (positive or negative), segmentation
        # label vector edge coordinates
        self.imgfiles = []
        self.labels = []
        self.seglabels = []

        # list of indices of positive and negative images
        self.positive_indices = []
        self.negative_indices = []

        # read in segmentation label files
        seglabels = []
        segfile_lookup = {}
        for i, seglabelfile in enumerate(os.listdir(seglabeldir)):
            segdata = json.load(open(os.path.join(seglabeldir,
                                                  seglabelfile), 'r'))
            seglabels.append(segdata)
            segfile_lookup[
                "-".join(segdata['data']['image'].split('-')[1:]).replace(
                    '.png', '.tif')] = i

        # read in image file names for positive images
        idx = 0
        for root, dirs, files in os.walk(datadir):
            for filename in files:
                if not filename.endswith('.tif'):
                    continue
                if filename not in segfile_lookup.keys():
                    continue
                polygons = []
                for completions in seglabels[segfile_lookup[filename]][
                    'completions']:
                    for result in completions['result']:
                        polygons.append(
                            np.array(result['value']['points'] +
                                     [result['value']['points'][0]]) * 1.2)
                        # factor of 1.2 necessary to scale edge coordinates
                        # appropriately
                if 'positive' in root and polygons != []:
                    self.labels.append(True)
                    self.positive_indices.append(idx)
                    self.imgfiles.append(os.path.join(root, filename))
                    self.seglabels.append(polygons)
                    idx += 1

        # add as many negative example images
        for root, dirs, files in os.walk(datadir):
            for filename in files:
                if not filename.endswith('.tif'):
                    continue
                if idx >= len(self.positive_indices)*2:
                    break
                if 'negative' in root:
                    self.labels.append(False)
                    self.negative_indices.append(idx)
                    self.imgfiles.append(os.path.join(root, filename))
                    self.seglabels.append([])
                    idx += 1

        # turn lists into arrays
        self.imgfiles = np.array(self.imgfiles)
        self.labels = np.array(self.labels)
        self.positive_indices = np.array(self.positive_indices)
        self.negative_indices = np.array(self.negative_indices)

        # increase data set size by factor `mult`
        if mult > 1:
            self.imgfiles = np.array([*self.imgfiles] * mult)
            self.labels = np.array([*self.labels] * mult)
            self.positive_indices = np.array([*self.positive_indices] * mult)
            self.negative_indices = np.array([*self.negative_indices] * mult)
            self.seglabels = self.seglabels * mult
            

    def __len__(self):
        """Returns length of data set."""
        return len(self.imgfiles)


    def __getitem__(self, idx):
        """Read in image data, preprocess, build segmentation mask, and apply
        transformations."""

        # read in image data
        imgfile = rio.open(self.imgfiles[idx])
        imgdata = np.array([imgfile.read(i) for i in
                            [1,2,3,4,5,6,7,8,9,10,12,13]])
        # skip band 11 (Sentinel-2 Band 10, Cirrus) as it does not contain
        # useful information in the case of Level-2A data products

        # force image shape to be 120 x 120 pixels
        if imgdata.shape[1] != 120:
            newimgdata = np.empty((12, 120, imgdata.shape[2]))
            newimgdata[:, :imgdata.shape[1], :] = imgdata[:,
                                                  :imgdata.shape[1], :]
            newimgdata[:, imgdata.shape[1]:, :] = imgdata[:,
                                                  imgdata.shape[1]-1:, :]
            imgdata = newimgdata
        if imgdata.shape[2] != 120:
            newimgdata = np.empty((12, 120, 120))
            newimgdata[:, :, :imgdata.shape[2]] = imgdata[:,
                                                  :, :imgdata.shape[2]]
            newimgdata[:, :, imgdata.shape[2]:] = imgdata[:,
                                                  :, imgdata.shape[2]-1:]
            imgdata = newimgdata

        # rasterize segmentation polygons
        fptdata = np.zeros(imgdata.shape[1:], dtype=np.uint8)
        polygons = self.seglabels[idx]
        shapes = []
        if len(polygons) > 0:
            for pol in polygons:
                try:
                    pol = Polygon(pol)
                    shapes.append(pol)
                except ValueError:
                    continue
            fptdata = rasterize(((g, 1) for g in shapes),
                                out_shape=fptdata.shape,
                                all_touched=True)

        sample = {'idx': idx,
                  'img': imgdata,
                  'fpt': fptdata,
                  'imgfile': self.imgfiles[idx]}

        # apply transformations
        if self.transform:
            sample = self.transform(sample)

        return sample


    def display(self, idx):
        """Helper method to display a given example from the data set with
        index `idx`. Only RGB channels are displayed, as well as the
        segmentation label outlines.

        :param idx: (int) image index to be displayed
        :param offset: (float) constant scaling offset (on a range [0,1])
        :param scaling: (float) scaling factor
        :return: `matplotlib.pyplot.figure` object
        """
        sample = self[idx]
        imgdata = sample['img']
        fptdata = sample['fpt']

        # scale image data
        imgdata = offset+scaling*(
            np.dstack([imgdata[3], imgdata[2], imgdata[1]])-
            np.min([imgdata[3], imgdata[2], imgdata[1]]))/ \
                (np.max([imgdata[3], imgdata[2], imgdata[1]])-
                 np.min([imgdata[3], imgdata[2], imgdata[1]]))

        f, ax = plt.subplots(1, 2, figsize=(6, 3))
        ax[0].imshow(imgdata)
        ax[1].imshow(fptdata)

        return f  

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        """
        :param sample: sample to be converted to Tensor
        :return: converted Tensor sample
        """

        out = {'idx': sample['idx'],
               'img': torch.from_numpy(sample['img'].copy()),
               'fpt': torch.from_numpy(sample['fpt'].copy()),
               'imgfile': sample['imgfile']}

        return out

class Normalize(object):
    """Normalize pixel values to zero mean and range [-1, +1] measured in
    standard deviations."""
    def __init__(self):
        """
        :param size: edge length of quadratic output size
        """
        self.channel_means = np.array(
            [809.2, 900.5, 1061.4, 1091.7, 1384.5, 1917.8,
             2105.2, 2186.3, 2224.8, 2346.8, 1901.2, 1460.42])
        self.channel_stds = np.array(
            [441.8, 624.7, 640.8, 718.1, 669.1, 767.5,
             843.3, 947.9, 882.4, 813.7, 716.9, 674.8])

    def __call__(self, sample):
        """
        :param sample: sample to be normalized
        :return: normalized sample
        """

        sample['img'] = (sample['img']-self.channel_means.reshape(
            sample['img'].shape[0], 1, 1))/self.channel_stds.reshape(
            sample['img'].shape[0], 1, 1)

        return sample

class Randomize(object):
    """Randomize image orientation including rotations by integer multiples of
       90 deg, (horizontal) mirroring, and (vertical) flipping."""

    def __call__(self, sample):
        """
        :param sample: sample to be randomized
        :return: randomized sample
        """
        imgdata = sample['img']
        fptdata = sample['fpt']

        # mirror horizontally
        mirror = np.random.randint(0, 2)
        if mirror:
            imgdata = np.flip(imgdata, 2)
            fptdata = np.flip(fptdata, 1)
        # flip vertically
        flip = np.random.randint(0, 2)
        if flip:
            imgdata = np.flip(imgdata, 1)
            fptdata = np.flip(fptdata, 0)
        # rotate by [0,1,2,3]*90 deg
        rot = np.random.randint(0, 4)
        imgdata = np.rot90(imgdata, rot, axes=(1,2))
        fptdata = np.rot90(fptdata, rot, axes=(0,1))

        return {'idx': sample['idx'],
                'img': imgdata.copy(),
                'fpt': fptdata.copy(),
                'imgfile': sample['imgfile']}

class RandomCrop(object):
    """Randomly crop 90x90 pixel image (from 120x120)."""

    def __call__(self, sample):
        """
        :param sample: sample to be cropped
        :return: randomized sample
        """
        imgdata = sample['img']

        x, y = np.random.randint(0, 30, 2)

        return {'idx': sample['idx'],
                'img': imgdata.copy()[:, y:y+90, x:x+90],
                'fpt': sample['fpt'].copy()[y:y+90, x:x+90],
                'imgfile': sample['imgfile']}


def create_dataset(*args, apply_transforms=True, **kwargs):
    """Create a dataset; uses same input parameters as PowerPlantDataset.
    :param apply_transforms: if `True`, apply available transformations
    :return: data set"""
    if apply_transforms:
        data_transforms = transforms.Compose([
            Normalize(),
            Randomize(),
            RandomCrop(),
            ToTensor()
           ])
    else:
        data_transforms = None

    data = SmokePlumeSegmentationDataset(*args, **kwargs,
                                         transform=data_transforms)

    return data
