import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn

from mxnet.contrib.symbol import Proposal, ROIAlign

class ProposalNetwork(gluon.nn.HybridBlock):
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


class ProposalLayer(gluon.nn.HybridBlock):
	def __init__(self, scales, ratios, Nt=0.7, rpn_post_nms_top_n=5):
		super(ProposalLayer, self).__init__()
		self.scales = scales
		self.ratios = ratios

		self.Nt = Nt
		self.rpn_post_nms_top_n = rpn_post_nms_top_n

		with self.name_scope():
			self.im_info = self.params.get('im_info', shape=(1,3), init=mx.init.Constant([[1024,1024,1]]), allow_deferred_init=True)
	
	def region_proposal(self, data, im_info):
		proposal = Proposal(cls_prob=data[0], bbox_pred=data[1], im_info=im_info,
							threshold=self.Nt, rpn_post_nms_top_n=self.rpn_post_nms_top_n,
							rpn_pre_nms_top_n=50, output_score=False, iou_loss=False,
							feature_stride=32, scales=self.scales, ratios=self.ratios)
		return proposal, im_info
	
	def hybrid_forward(self, F, cls_scores, bbox_pred, im_info):
		cls_scores = F.reshape(cls_scores, shape=(0,1,18,32,32))
		bbox_pred = F.reshape(bbox_pred, shape=(0,1,36,32,32))
		proposals, _ = F.contrib.foreach(self.region_proposal, [cls_scores, bbox_pred], im_info)
		return proposals

class ROIAlignmentLayer(gluon.nn.HybridBlock):
	def __init__(self, pooled_size, spatial_scale=0.03125):
		super(ROIAlignmentLayer, self).__init__()
		self.pooled_size = pooled_size
		self.spatial_scale = spatial_scale
	
	def roi_alignment(self, data, _):
		region = ROIAlign(data=data[0], rois=data[1],
							pooled_size=self.pooled_size, spatial_scale=self.spatial_scale)
		return region, _

	def hybrid_forward(self, F, feature_map, rois):
		feature_map = F.reshape(feature_map, shape=(0,1,512,32,32))
		rois = F.reshape(rois, shape=(0,5,5))
		regions, _ = F.contrib.foreach(self.roi_alignment, [feature_map, rois], [])
		return regions