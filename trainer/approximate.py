from mxnet import nd,gpu,init,gluon,autograd

from data.dicom import DICOMFolderDataset
from network.rcnn import RoentgenFasterRCNN

from mxboard import SummaryWriter

# devices
ctx = gpu(0)

pneumothorax = RoentgenFasterRCNN()
pneumothorax.hybridize()

Nt = 0.7
Pt = 0.5

# Kaiming initialization from uniform [-c,c] c = sqrt(2/Nin) 
pneumothorax.collect_params().initialize(init.Xavier(factor_type='in', magnitude=0.44444), ctx=ctx)

train_data = DICOMFolderDataset('siim-acr-data/train-pneumothorax/**/**/*.dcm', 'siim-acr-data/train.csv')
train_loader = gluon.data.DataLoader(train_data, 16, shuffle=True, num_workers=1)

valid_data = DICOMFolderDataset('siim-acr-data/dev-pneumothorax/**/**/*.dcm', 'siim-acr-data/train-dev.csv')
valid_loader = gluon.data.DataLoader(valid_data, 16, shuffle=False, num_workers=1)


import math
import itertools
import numpy as np

stride = 32

sizes = [0.25,0.15,0.05]
ratios = [2,1,0.5]

archetypes = list(itertools.product(sizes,ratios))
anchor_boxes = nd.array([(size,size*ratio) for size, ratio in archetypes], dtype=np.float32, ctx=ctx)

dx = range(int(stride/2),int(1024),stride)
dy = range(int(stride/2),int(1024),stride)

anchor_points = list(itertools.product(dy,dx))
anchor_points = nd.array(anchor_points, dtype=np.float32, ctx=ctx)

anchor_points = anchor_points.transpose()/1024
anchor_points[[0, 1]] = anchor_points[[1, 0]]
anchor_points = anchor_points.reshape(2,32,32)


"""Approximate Joint Training"""

epochs = 350

import pandas as pd
from tqdm import tqdm

from trainer.iou import box_iou

lambd = 10
Ncls = 12
Nreg = 1024

pos_weight = nd.array([1.05],ctx=ctx)
box_weight = nd.array([1.5,1.5,1.5,1.5],ctx=ctx)

huber_loss = gluon.loss.HuberLoss(rho=1, weight=5E+1)
binary_cross_entropy = gluon.loss.SigmoidBinaryCrossEntropyLoss(weight=2E+2)

trainer = gluon.Trainer(pneumothorax.collect_params(), 'adam', {'learning_rate': 1E-5})

with SummaryWriter(logdir='./logs/pneumothorax-rpn') as log:
    for epoch in range(epochs):
        with tqdm(total=int(len(train_data)/16+len(valid_data)/16), desc='Approximate Joint Training (RPN): Epoch {}'.format(epoch)) as pg:
            cumulated_loss = 0.0
            cumulated_bce = 0.0
            cumulated_huber = 0.0

            for i, (data, labels) in enumerate(train_loader):
                data = data.as_in_context(ctx)
                labels = labels.as_in_context(ctx)

                batch_size = data.shape[0]
                X = data.reshape((batch_size, 1, 1024, 1024))

                p = nd.broadcast_to(anchor_points.reshape(1,1,2,32,32), (batch_size,9,2,32,32))
                s = nd.broadcast_to(anchor_boxes.reshape(1,9,2,1,1),(batch_size,9,2,32,32)) # aw = ah = s*sqrt(r)
                A = nd.concat(p,s,dim=2)

                G = nd.broadcast_to(labels.reshape(batch_size,1,4,1,1),(batch_size,9,4,32,32))

                IOU = box_iou(A,G)

                # fg/bg threshold
                p = IOU > Nt

                # max IOU if there is no overlap > Nt
                m = IOU.max(axis=(1,2,3))

                m = nd.where(m<=Nt,m,-1*m)
                m = nd.where(m==0,m-1,m)

                for i in range(batch_size):
                    p[i] = p[i] + (IOU[i]==m[i])

                # calculation mask for the regression loss
                p_reg = nd.broadcast_to(p.reshape(batch_size,9,1,32,32), (batch_size,9,4,32,32))

                box_weight = nd.broadcast_to(box_weight.reshape(1,1,4,1,1), (batch_size,9,4,32,32))

                nd.waitall()

                with autograd.record():
                    # proposal generation
                    s,b = pneumothorax(X)
                    b = b.reshape(batch_size,9,4,32,32)

                    # offset from anchor
                    r = G - A

                    # on a pixel scale
                    gt_offset = nd.multiply(r,p_reg)*1024
                    bbox_offset = nd.multiply(b,p_reg)*1024

                    # multi-task loss
                    Lcls = binary_cross_entropy(s,p,pos_weight)
                    Lreg = huber_loss(bbox_offset,gt_offset,box_weight)

                    L = Lcls/Ncls + lambd*Lreg/Nreg
                
                L.backward()
                trainer.step(data.shape[0])

                cumulated_bce += Lcls.mean().asscalar()
                cumulated_huber += Lreg.mean().asscalar()
                cumulated_loss += L.mean().asscalar()

                pg.update()

            print(cumulated_loss/(len(valid_data)/16))
            print(cumulated_bce/(len(valid_data)/16))
            print(cumulated_huber/(len(valid_data)/16))
            log.add_scalar(tag='loss', value=(cumulated_loss/(len(valid_data)/16)), global_step=epoch)
            log.add_scalar(tag='bce', value=(cumulated_bce/(len(valid_data)/16)), global_step=epoch)
            log.add_scalar(tag='huber', value=(cumulated_huber/(len(valid_data)/16)), global_step=epoch)
            
            nd.waitall()


            tp = 0
            fp = 0
            fn = 0

            n = 0
            cumulated_error = 0

            for i, (data, labels) in enumerate(valid_loader):
                data = data.as_in_context(ctx)
                labels = labels.as_in_context(ctx)

                batch_size = data.shape[0]
                X = data.reshape((batch_size, 1, 1024, 1024))

                p = nd.broadcast_to(anchor_points.reshape(1,1,2,32,32), (batch_size,9,2,32,32))
                s = nd.broadcast_to(anchor_boxes.reshape(1,9,2,1,1),(batch_size,9,2,32,32)) # aw = ah = s*sqrt(r)
                A = nd.concat(p,s,dim=2)

                G = nd.broadcast_to(labels.reshape(batch_size,1,4,1,1),(batch_size,9,4,32,32))

                IOU = box_iou(A,G)

                # fg/bg threshold
                p = IOU > Nt
                
                # max IOU if there is no overlap > Nt
                m = IOU.max(axis=(1,2,3))

                m = nd.where(m<=Nt,m,-1*m)
                m = nd.where(m==0,m-1,m)

                for i in range(batch_size):
                    p[i] = p[i] + (IOU[i]==m[i])

                p_reg = nd.broadcast_to(p.reshape(batch_size,9,1,32,32), (batch_size,9,4,32,32))

                nd.waitall()


                # predict regions of interest
                s,b = pneumothorax(X)
                s = nd.sigmoid(s)
                b = b.reshape(1,9,4,32,32)

                # apply offsets
                B = A + b

                # calculate the confusion matrix for the classifier
                tp += nd.broadcast_logical_and(s > Pt, p).sum().asscalar()
                fp += nd.broadcast_logical_and(s > Pt, nd.logical_not(p)).sum().asscalar()
                fn += nd.broadcast_logical_and(s <= Pt, p).sum().asscalar()

                # mean absolute error for bounding box regression
                Y = nd.multiply(p_reg,G)
                Y_hat = nd.multiply(p_reg,B)

                cumulated_error += nd.abs(Y - Y_hat).sum().asscalar()
                n += (Y > 0).sum().asscalar()

                pg.update()

            log.add_scalar(tag='precision', value=(tp/((tp+fp) if tp>0 else 1)), global_step=epoch)
            log.add_scalar(tag='recall', value=(tp/((tp+fn) if tp>0 else 1)), global_step=epoch)
            log.add_scalar(tag='mae', value=(cumulated_error/n if n>0 else 0), global_step=epoch)

            pneumothorax.export("roentgen-pneumothorax-rpn", epoch=epoch)