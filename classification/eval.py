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
# This file contains routines for the evaluation of the trained model
# based on the test data set.

import os
import numpy as np
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt

from model import *
from data import create_dataset

np.random.seed(100)
torch.manual_seed(100)

# load data
batch_size = 10 # 1 to create diagnostic images, any value otherwise
testdata = create_dataset(datadir='/home/mommermi/hsg/data/smoke_emission'
                                  '/2020Neurips_dataset/classification/test')#path/to/test/data')
all_dl = DataLoader(testdata, batch_size=batch_size, shuffle=True)
progress = tqdm(enumerate(all_dl), total=len(all_dl))

# load model
model.load_state_dict(torch.load(
    'classification.model', map_location=torch.device('cpu')))
model.eval()


# implant hooks for resnet layers
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.relu.register_forward_hook(get_activation('conv1'))
model.layer1.register_forward_hook(get_activation('layer1'))
model.layer2.register_forward_hook(get_activation('layer2'))
model.layer3.register_forward_hook(get_activation('layer3'))
model.layer4.register_forward_hook(get_activation('layer4'))

# run through test data set
true = 0
false = 0
for i, batch in progress:
    x, y = batch['img'].float().to(device), batch['lbl'].float().to(device)

    output = model(x)
    prediction = 1 if output[0] > 0 else 0

    if prediction == 1 and y[0] == 1:
        res = 'true_pos'
        true += 1
    elif prediction == 0 and y[0] == 0:
        res = 'true_neg'
        true += 1
    elif prediction == 0 and y[0] == 1:
        res = 'false_neg'
        false += 1
    elif prediction == 1 and y[0] == 0:
        res = 'false_pos'
        false += 1

    if batch_size == 1:

        # create plot
        f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(1, 3))

        # rgb plot
        ax1.imshow(0.2+1.5*(np.dstack([x[0][3], x[0][2], x[0][1]])-
                    np.min([x[0][3].numpy(),
                            x[0][2].numpy(),
                            x[0][1].numpy()]))/
                   (np.max([x[0][3].numpy(),
                            x[0][2].numpy(),
                            x[0][1].numpy()])-
                    np.min([x[0][3].numpy(),
                            x[0][2].numpy(),
                            x[0][1].numpy()])),
                   origin='upper')
        ax1.set_title({'true_pos': 'True Positive',
                       'true_neg': 'True Negative',
                       'false_pos': 'False Positive',
                       'false_neg': 'False Negative'}[res],
                      fontsize=8)
        ax1.set_ylabel('RGB', fontsize=8)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # false color plot
        # RGB = (Aerosols; Water Vapor; SWIR1)
        ax2.imshow(0.2+(np.dstack([x[0][0], x[0][9], x[0][10]])-
                    np.min([x[0][0].numpy(),
                            x[0][9].numpy(),
                            x[0][10].numpy()]))/
                   (np.max([x[0][0].numpy(),
                            x[0][9].numpy(),
                            x[0][10].numpy()])-
                    np.min([x[0][0].numpy(),
                            x[0][9].numpy(),
                            x[0][10].numpy()])),
                   origin='upper')
        ax2.set_ylabel('False Color', fontsize=8)
        ax2.set_xticks([])
        ax2.set_yticks([])

        # layer2 activations plot
        map_layer2 = ax3.imshow(activation['layer2'].sum(axis=(0, 1)),
                                vmin=50, vmax=150)
        ax3.set_ylabel('Layer2', fontsize=8)
        ax3.set_xticks([])
        ax3.set_yticks([])

        f.subplots_adjust(0.05, 0.02, 0.95, 0.9, 0.05, 0.05)

        plt.savefig((res+os.path.split(batch['imgfile'][0])[1]).\
                    replace('.tif', '_eval.png').replace(':', '_'),
                    tight_layout=True, dpi=200)
        plt.close()

print('test set accuracy:', true/(true+false))

