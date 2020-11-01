from mxnet import gluon
from mxnet.gluon import nn

class ProposalNetwork(gluon.nn.HybridBlock):
	"""Proposal network calculating confidence level scores and bounding box offsets.

		Two convolution layers simulateously regress region bounds and objectness scores.

    Parameters
    ----------
	anchor_points : sliding-window location
		arranged on a regular rectangular grid.
	num_anchors : k proposals
		predicting 2k scores and 4k offsets.


    """
	def __init__(self, num_channels, num_anchors=9, anchor_points=(32,32)):
		super(ProposalNetwork, self).__init__()
		self.num_anchors = num_anchors
		self.anchor_points = anchor_points

		self.conv = nn.Conv2D(num_channels, kernel_size=5, strides=1, padding=2)

		self.cls_score = nn.Conv2D(2*num_anchors, kernel_size=1, strides=1)
		self.bbox_pred = nn.Conv2D(4*num_anchors, kernel_size=1, strides=1)

	def hybrid_forward(self, F, x):
		y = self.conv(x)
		cls_scores = self.cls_score(y)
		bbox_offsets = self.bbox_pred(y)
		return cls_scores, bbox_offsets