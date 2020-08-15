import argparse

from trainer.iou import box_iou
from mxnet import nd,gpu,init,gluon,autograd

from data.dicom import DICOMFolderDataset
from network.rcnn import RoentgenFasterRCNN

from tqdm import tqdm
from mxboard import SummaryWriter

import math
import itertools
import numpy as np

def approximate(epochs, ctx, args):
    pneumothorax = RoentgenFasterRCNN()
    pneumothorax.hybridize()

    # Kaiming initialization from uniform [-c,c] c = sqrt(2/Nin) 
    pneumothorax.collect_params().initialize(init.Xavier(factor_type='in', magnitude=0.44444), ctx=ctx)

    train_data = DICOMFolderDataset('siim-acr-data/train-pneumothorax/**/**/*.dcm', 'siim-acr-data/train.csv')
    train_loader = gluon.data.DataLoader(train_data, 16, shuffle=True, num_workers=1)

    valid_data = DICOMFolderDataset('siim-acr-data/dev-pneumothorax/**/**/*.dcm', 'siim-acr-data/train-dev.csv')
    valid_loader = gluon.data.DataLoader(valid_data, 16, shuffle=False, num_workers=1)


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

    pos_weight = nd.array([args.beta],ctx=ctx)
    box_weight = nd.array([args.gamma,args.gamma,args.gamma,args.gamma],ctx=ctx)

    rpn_huber_loss = gluon.loss.HuberLoss(rho=args.rho, weight=5E+1)
    rpn_binary_cross_entropy = gluon.loss.SigmoidBinaryCrossEntropyLoss(weight=2E+2)

    trainer = gluon.Trainer(pneumothorax.collect_params(), 'adam', {'learning_rate': args.learning_rate})

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
                    p = IOU > args.iou_threshold

                    # max IOU if there is no overlap > threshold
                    m = IOU.max(axis=(1,2,3))

                    m = nd.where(m<=args.iou_threshold,m,-1*m)
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
                        Lcls = rpn_binary_cross_entropy(s,p,pos_weight)
                        Lreg = rpn_huber_loss(bbox_offset,gt_offset,box_weight)

                        L = Lcls/args.Ncls + args.balance*Lreg/args.Nreg
                    
                    L.backward()
                    trainer.step(data.shape[0])

                    cumulated_bce += Lcls.mean().asscalar()
                    cumulated_huber += Lreg.mean().asscalar()
                    cumulated_loss += L.mean().asscalar()

                    pg.update()

                log.add_scalar(tag='rpn_loss', value=(cumulated_loss/(len(valid_data)/16)), global_step=epoch)
                log.add_scalar(tag='rpn_bce', value=(cumulated_bce/(len(valid_data)/16)), global_step=epoch)
                log.add_scalar(tag='rpn_huber', value=(cumulated_huber/(len(valid_data)/16)), global_step=epoch)
                
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
                    p = IOU > args.iou_threshold
                    
                    # max IOU if there is no overlap > Nt
                    m = IOU.max(axis=(1,2,3))

                    m = nd.where(m<=args.iou_threshold,m,-1*m)
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
                    tp += nd.broadcast_logical_and(s > args.nms_threshold, p).sum().asscalar()
                    fp += nd.broadcast_logical_and(s > args.nms_threshold, nd.logical_not(p)).sum().asscalar()
                    fn += nd.broadcast_logical_and(s <= args.nms_threshold, p).sum().asscalar()

                    # mean absolute error for bounding box regression
                    Y = nd.multiply(p_reg,G)
                    Y_hat = nd.multiply(p_reg,B)

                    cumulated_error += nd.abs(Y - Y_hat).sum().asscalar()
                    n += (Y > 0).sum().asscalar()

                    pg.update()

                log.add_scalar(tag='rpn_precision', value=(tp/((tp+fp) if tp>0 else 1)), global_step=epoch)
                log.add_scalar(tag='rpn_recall', value=(tp/((tp+fn) if tp>0 else 1)), global_step=epoch)
                log.add_scalar(tag='rpn_mae', value=(cumulated_error/n if n>0 else 0), global_step=epoch)

                pneumothorax.export("roentgen-pneumothorax-rpn", epoch=epoch)

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    # devices
    ctx = gpu(0)

    parser.add_argument('--epochs', '-n', type=int)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1E-5)
    parser.add_argument('--beta', '-beta', type=float, default=1.7, help='Value > 1 increases recall for the RPN')
    parser.add_argument('--gamma', '-gamma', type=float, default=1.5)
    parser.add_argument('--rho', '-rho', type=float, default=1)
    parser.add_argument('--balance', '-lambda', type=int, default=10, help='Balance between classification and regression loss')
    parser.add_argument('--Ncls', '-Ncls', type=int, default=16)
    parser.add_argument('--Nreg', '-Nreg', type=int, default=1024)
    parser.add_argument('--nms_threshold', '-nms', type=float, default=0.5)
    parser.add_argument('--iou_threshold', type=float, default=0.7, help='Threshold for an anchor box to be picked as forground')

    args = parser.parse_args()
    approximate(args.epochs, ctx, args)