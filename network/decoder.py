import itertools
import numpy as np
import mxnet as mx
from mxnet import autograd,gluon,nd


class AnchorBoxDecoder(gluon.nn.HybridBlock):
    """Decode bounding boxes training target from ProposalNetwork offsets.

    Returned bounding boxes are in corner format: (xmin, ymin, xmax, ymax).

    Parameters
    ----------
    map_stride : resolution of the feature map
        anchor points are centered on a regular grid.
    sizes : width
        percentage of the full size of the 1024x1024px image.
    ratios : height to width
        height of the anchor box is scaled by the given factor.

    """
    def __init__(self, map_stride, iou_threshold=0.7, cls_threshold=0.5, sizes=[0.25,0.15,0.05], ratios=[2,1,0.5], rpn_head=False, iou_output=False):
        super(AnchorBoxDecoder, self).__init__()
        self.rpn_head = rpn_head
        self.iou_output = iou_output
        self.iou_threshold = iou_threshold
        self.cls_threshold = cls_threshold
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
    
    def center_format(self, F, bbox):
        """
        corner format: [xmin, ymin, xmax, ymax]
        conversion to center format:
            w = xmax - xmin
            h = ymax - ymin
            x = xmin + w/2
            y = ymin + h/2
        """
        w = F.slice_axis(bbox, axis=2, begin=2, end=3) - F.slice_axis(bbox, axis=2, begin=0, end=1)
        h = F.slice_axis(bbox, axis=2, begin=3, end=4) - F.slice_axis(bbox, axis=2, begin=1, end=2)
        x = F.slice_axis(bbox, axis=2, begin=0, end=1) + w/2
        y = F.slice_axis(bbox, axis=2, begin=1, end=2) + h/2

        return F.concat(x,y,w,h,dim=2)



    def corner_format(self, F, bbox):
        """
        conversion to corner format:
            ymin = y - h/2
            xmin = x - w/2
            ymax = y + h/2
            xmax = x + w/2
        """
        ymin = F.slice_axis(bbox, axis=2, begin=1, end=2) - F.slice_axis(bbox, axis=2, begin=3, end=4)/2
        xmin = F.slice_axis(bbox, axis=2, begin=0, end=1) - F.slice_axis(bbox, axis=2, begin=2, end=3)/2
        ymax = F.slice_axis(bbox, axis=2, begin=1, end=2) + F.slice_axis(bbox, axis=2, begin=3, end=4)/2
        xmax = F.slice_axis(bbox, axis=2, begin=0, end=1) + F.slice_axis(bbox, axis=2, begin=2, end=3)/2

        return ymin, xmin, ymax, xmax
    
    def box_iou(self,F,A,G):

        # convert from center to corner format
        Gymin, Gxmin, Gymax, Gxmax = self.corner_format(F, G)
        Aymin, Axmin, Aymax, Axmax = self.corner_format(F, A)

        # Ai
        dx = F.minimum(Gxmax,Axmax) - F.maximum(Gxmin,Axmin)
        dy = F.minimum(Gymax,Aymax) - F.maximum(Gymin,Aymin)

        #if dx or dy is negative no intersection else dx hadamard dy
        Ai = F.broadcast_mul(F.relu(dx),F.relu(dy))
    
        Au = F.broadcast_mul(F.slice_axis(A, axis=2, begin=2, end=3),F.slice_axis(A, axis=2, begin=3, end=4)) + F.broadcast_mul(F.slice_axis(G, axis=2, begin=2, end=3),F.slice_axis(G, axis=2, begin=3, end=4)) - Ai

        return F.relu(F.broadcast_div(Ai,Au))


    def hybrid_forward(self, F, rpn_cls_scores, rpn_bbox_offsets, labels=None, **kwargs):
        # split anchor and offset predictions
        rpn_bbox_offsets = F.reshape(rpn_bbox_offsets,(0,self.num_anchors,4,32,32))
        # broadcast across all boxes 
        points = F.broadcast_to(F.reshape(kwargs["anchor_points"],(1,1,2,32,32)), (1,self.num_anchors,2,32,32))
        # broadcast over all points
        sizes = F.broadcast_to(F.reshape(kwargs["anchor_boxes"],(1,9,2,1,1)), (1,self.num_anchors,2,32,32))
        # broadcast to batch
        anchors = F.concat(points,sizes,dim=2)
        rpn_bbox_anchors = F.broadcast_like(anchors, rpn_bbox_offsets)

        # construct network for training region proposals
        if self.rpn_head:
            # broadcast to all sliding window positions
            ground_truth = F.broadcast_to(F.reshape(labels,(0,1,4,1,1)), (0,9,4,32,32))

            # offsets are computed for center format
            ground_truth = self.center_format(F, ground_truth)

            # intersection over union
            ious = self.box_iou(F,rpn_bbox_anchors, ground_truth)

            # select anchor boxes as fg/bg
            mask = ious > self.iou_threshold

            # maximum IOU anchor box along all sliding window locations
            attention = ious.max(axis=(1,2,3,4))
            # ignore maximum smaller than the threshold
            attention = F.where(attention<=self.iou_threshold, attention, -1*attention)
            # ignore zero maximum
            attention = F.where(attention==0, attention-1, attention)

            # select maximum IOU if there is no overlap bigger than the threshold
            attention_mask = mask + F.broadcast_equal(ious, F.reshape(attention,(0,1,1,1,1)))

            # create a mask for iou selection
            calculation_mask = F.broadcast_like(attention_mask, rpn_bbox_offsets)

            # mask out invalid values
            rpn_gt_offsets = F.broadcast_mul(ground_truth - rpn_bbox_anchors, calculation_mask)
            rpn_bbox_offsets = F.broadcast_mul(rpn_bbox_offsets, calculation_mask)
            rpn_bbox_anchors = F.broadcast_mul(rpn_bbox_anchors, calculation_mask)

            if self.iou_output:
                # apply predicted offset to anchors
                rpn_bbox_pred = rpn_bbox_anchors + rpn_bbox_offsets

                rpn_bbox_ious = self.box_iou(F,rpn_bbox_pred, ground_truth)

                return rpn_bbox_anchors, rpn_bbox_offsets, rpn_gt_offsets, ground_truth, attention_mask, rpn_bbox_ious
            else:
                return rpn_bbox_anchors, rpn_bbox_offsets, rpn_gt_offsets, attention_mask
        
        # construct network in inference mode
        else:
            # apply predictions to anchors
            rpn_bbox_pred = rpn_bbox_anchors + rpn_bbox_offsets

            # [id, a, s, x, y]
            F.softmax(rpn_cls_scores, axis=2)

            # slice ymin/xmin/ymay/xmax = [id, a, 1, 32, 32] along axis 2
            ymin, xmin, ymax, xmax = self.corner_format(F, rpn_bbox_pred)

            # TODO: extract rois into format roi=[id, score, xmin, ymin, xmax, ymax]

            # feed rpn_cls_scores for this
            #mx.sym.softmax(rpn_cls,axis=)
            return rpn_bbox_rois
