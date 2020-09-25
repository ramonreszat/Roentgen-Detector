import itertools
import numpy as np
import mxnet as mx
from mxnet import autograd,gluon,nd


class AnchorBoxDecoder(gluon.nn.HybridBlock):
    """Decode bounding boxes training target from ProposalNetwork offsets.

    Returned bounding boxes are using corner type: `x_{min}, y_{min}, x_{max}, y_{max}`.

    Parameters
    ----------
    map_stride : resolution of the feature map
        anchor points are centered on a regular grid.
    sizes : width
        percentage of the full size of the 1024x1024px image.
    ratios : height to width
        height of the anchor box is scaled by the given factor.

    """
    def __init__(self, map_stride, iou_threshold=0.7, sizes=[0.25,0.15,0.05], ratios=[2,1,0.5]):
        super(AnchorBoxDecoder, self).__init__()
        self.iou_threshold = iou_threshold
        self.num_anchors = len(sizes) * len(ratios)

        archetypes = list(itertools.product(sizes,ratios))
        anchor_boxes = np.array([(size,size*ratio) for size, ratio in archetypes], dtype=np.float32)

        dx = range(int(map_stride/2),int(1024),map_stride)
        dy = range(int(map_stride/2),int(1024),map_stride)

        anchor_points = list(itertools.product(dy,dx))
        anchor_points = nd.array(anchor_points, dtype=np.float32)

        anchor_points = anchor_points.transpose()/1024
        anchor_points[[0, 1]] = anchor_points[[1, 0]]
        anchor_points = anchor_points.reshape(2,32,32)

        with self.name_scope():
            self.anchor_points = self.params.get_constant('anchor_points', anchor_points)
            self.anchor_boxes = self.params.get_constant('anchor_boxes', anchor_boxes)
    
    def box_iou(self,F,A,G):

        # conversion to min/max rectangle span
        #     ymin = y - h/2
        #     xmin = x - w/2
        #     ymax = y + h/2
        #     xmax = x + w/2
        Gymin = G[:,:,1,:,:] - G[:,:,3,:,:]/2
        Gxmin = G[:,:,0,:,:] - G[:,:,2,:,:]/2
        Gymax = G[:,:,1,:,:] + G[:,:,3,:,:]/2
        Gxmax = G[:,:,0,:,:] + G[:,:,2,:,:]/2

        Aymin = A[:,:,1,:,:] - A[:,:,3,:,:]/2
        Axmin = A[:,:,0,:,:] - A[:,:,2,:,:]/2
        Aymax = A[:,:,1,:,:] + A[:,:,3,:,:]/2
        Axmax = A[:,:,0,:,:] + A[:,:,2,:,:]/2

        # Ai
        dx = F.minimum(Gxmax,Axmax) - F.maximum(Gxmin,Axmin)
        dy = F.minimum(Gymax,Aymax) - F.maximum(Gymin,Aymin)

        #if dx or dy is negative no intersection else dx hadamard dy
        Ai = F.multiply(F.relu(dx),F.relu(dy))
    

        # Au = s^2 + wh - Ai > 0
        Au = F.multiply(A[:,:,2,:,:],A[:,:,3,:,:]) + F.multiply(G[:,:,2,:,:],G[:,:,3,:,:]) - Ai

        return F.relu(F.divide(Ai,Au))
    
    def and_equals(self, data, _):
        return data[0] + (data[1]==data[2]), _


    def hybrid_forward(self, F, bbox_offsets, anchor_points, anchor_boxes, labels=None):
        bbox_offsets = F.reshape(bbox_offsets,(0,self.num_anchors,4,32,32))
        # broadcast across all boxes 
        points = F.broadcast_to(F.reshape(anchor_points,(1,1,2,32,32)), (1,self.num_anchors,2,32,32))
        # broadcast over all points
        sizes = F.broadcast_to(F.reshape(anchor_boxes,(1,9,2,1,1)), (1,self.num_anchors,2,32,32))
        # broadcast to batch
        anchors = F.concat(points,sizes,dim=2)
        A = F.broadcast_like(anchors, bbox_offsets)

        if autograd.is_training:
            # broadcast to all sliding window positions
            ground_truth = F.broadcast_to(F.reshape(labels,1,1,4,1,1), (1,9,4,32,32))
            # broadcast to batch
            G = F.broadcast_like(ground_truth, bbox_offsets)

            # intersection over union
            ious = self.box_iou(F,A,G)

            # select anchor boxes as fg/bg
            mask = ious > self.iou_threshold

            # maximum IOU anchor box along all sliding window locations
            attention = ious.max(axis=(1,2,3))
            # ignore maximum smaller than the threshold
            attention = F.where(attention<=self.iou_threshold,attention,-1*attention)
            # ignore zero maximum
            attention = F.where(attention==0,attention-1,attention)

            # select maximum IOU if there is no overlap bigger than the threshold
            mask, _ = F.contrib.foreach(self.and_equals, [mask, iou, attention], [])

        else:
            # apply predictions to anchors
            #TODO: anchor points are in center format
            return A + bbox_offsets
