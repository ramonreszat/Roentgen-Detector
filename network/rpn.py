from mxnet import nd, gluon
from mxnet.gluon import nn

class RegionProposal(gluon.nn.HybridBlock):
	def __init__(self, num_channels, num_anchors=9, anchor_points=(32,32)):
		super(RegionProposal, self).__init__()
		self.num_anchors = num_anchors
		self.anchor_points = anchor_points

		self.conv = nn.Conv2D(num_channels, kernel_size=5, strides=1, padding=2)

		self.cls_score = nn.Conv2D(num_anchors, kernel_size=1, strides=1)
		self.bbox_pred = nn.Conv2D(4*num_anchors, kernel_size=1, strides=1)

	def hybrid_forward(self, F, x):
		y = self.conv(x)

		s = self.cls_score(y)
		b = self.bbox_pred(y)
		
		return s,b