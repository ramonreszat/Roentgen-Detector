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
    
    def box_iou(self,A,G):

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
        dx = nd.minimum(Gxmax,Axmax) - nd.maximum(Gxmin,Axmin)
        dy = nd.minimum(Gymax,Aymax) - nd.maximum(Gymin,Aymin)

        #if dx or dy is negative no intersection else dx hadamard dy
        Ai = nd.multiply(nd.relu(dx),nd.relu(dy))
    

        # Au = s^2 + wh - Ai > 0
        Au = nd.multiply(A[:,:,2,:,:],A[:,:,3,:,:]) + nd.multiply(G[:,:,2,:,:],G[:,:,3,:,:]) - Ai

        return nd.relu(nd.divide(Ai,Au))


    def hybrid_forward(self, F, bbox_offsets, anchor_points, anchor_boxes, labels=None):
        bbox_offsets = bbox_offsets.reshape(0,self.num_anchors,4,32,32)
        #TODO: infer batch_size from input F
        points = nd.broadcast_to(anchor_points.reshape(1,1,2,32,32), (self.batch_size,self.num_anchors,2,32,32))
        sizes = nd.broadcast_to(anchor_boxes.reshape(1,9,2,1,1),(self.batch_size,self.num_anchors,2,32,32))
        A = nd.concat(points,sizes,dim=2)

        #TODO: anchor points are in center format
        if autograd.is_training:
            G = nd.broadcast_to(labels.reshape(self.batch_size,1,4,1,1),(self.batch_size,9,4,32,32))

            ious = self.box_iou(A,G)

            # fg/bg threshold
            p = ious > self.iou_threshold

            # max IOU if there is no overlap > threshold
            m = ious.max(axis=(1,2,3))

            m = nd.where(m<=self.iou_threshold,m,-1*m)
            m = nd.where(m==0,m-1,m)

            #TODO: symbol operation
            for i in range(batch_size):
                p[i] = p[i] + (ious[i]==m[i])

        else:
            # apply predictions to anchors
            return A + bbox_offsets
