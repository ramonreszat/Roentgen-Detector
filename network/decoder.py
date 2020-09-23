import itertools
import numpy as np
from mxnet import autograd,nd
from mxnet.gluon import HybridBlock

class AnchorBoxDecoder(HybridBlock):
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
    def __init__(self, batch_size, map_stride=32, sizes=[0.25,0.15,0.05], ratios=[2,1,0.5]):
        self.batch_size = batch_size
        archetypes = list(itertools.product(sizes,ratios))
        self.anchor_boxes = nd.array([(size,size*ratio) for size, ratio in archetypes], dtype=np.float32, ctx=ctx)

        dx = range(int(map_stride/2),int(1024),map_stride)
        dy = range(int(map_stride/2),int(1024),map_stride)

        #TODO: anchor points are in center format
        #TODO: pass as parameters to the forward function
        anchor_points = list(itertools.product(dy,dx))
        anchor_points = nd.array(anchor_points, dtype=np.float32, ctx=ctx)

        anchor_points = anchor_points.transpose()/1024
        anchor_points[[0, 1]] = anchor_points[[1, 0]]
        self.anchor_points = anchor_points.reshape(2,32,32)
        

    def hybrid_forward(self, bbox_offsets, labels, anchor_points, anchor_boxes):
        G = nd.broadcast_to(labels.reshape(self.batch_size,1,4,1,1),(self.batch_size,9,4,32,32))
        points = nd.broadcast_to(anchor_points.reshape(1,1,2,32,32), (self.batch_size,9,2,32,32))
        sizes = nd.broadcast_to(anchor_boxes.reshape(1,9,2,1,1),(self.batch_size,9,2,32,32)) # aw = ah = s*sqrt(r)
        anchors = nd.concat(points,sizes,dim=2)
        if autograd.is_training:
            return 
        else:
            return anchors + bbox_offsets
