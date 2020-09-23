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
    def __init__(self, map_stride, sizes=[0.25,0.15,0.05], ratios=[2,1,0.5]):
        super(AnchorBoxDecoder, self).__init__()

        #TODO: translate to numpy
        archetypes = list(itertools.product(sizes,ratios))
        anchor_boxes = nd.array([(size,size*ratio) for size, ratio in archetypes], dtype=np.float32, ctx=ctx)

        dx = range(int(map_stride/2),int(1024),map_stride)
        dy = range(int(map_stride/2),int(1024),map_stride)

        anchor_points = list(itertools.product(dy,dx))
        anchor_points = nd.array(anchor_points, dtype=np.float32, ctx=ctx)

        anchor_points = anchor_points.transpose()/1024
        anchor_points[[0, 1]] = anchor_points[[1, 0]]
        anchor_points = anchor_points.reshape(2,32,32)

        with self.name_scope():
            self.anchor_points = self.params.get('anchor_points',
                                                shape=anchor_points.shape,
                                                init=mx.init.Constant(anchor_points.asnumpy()),
                                                differentiable=False)
            self.anchor_boxes = self.params.get('anchor_boxes',
                                                shape=anchor_boxes.shape,
                                                init=mx.init.Constant(anchor_boxes.asnumpy()),
                                                differentiable=False)
        

    def hybrid_forward(self, b, anchor_points, anchor_boxes, labels=None):
        #TODO: infer batch_size from input
        G = nd.broadcast_to(labels.reshape(self.batch_size,1,4,1,1),(self.batch_size,9,4,32,32))
        points = nd.broadcast_to(anchor_points.reshape(1,1,2,32,32), (self.batch_size,9,2,32,32))
        sizes = nd.broadcast_to(anchor_boxes.reshape(1,9,2,1,1),(self.batch_size,9,2,32,32)) # aw = ah = s*sqrt(r)
        A = nd.concat(points,sizes,dim=2)

        #TODO: anchor points are in center format
        if autograd.is_training:
            return 
        else:
            return A + b
