from mxnet import gluon

from mxnet.contrib.symbol import ROIAlign

class ROIAlignmentLayer(gluon.nn.HybridBlock):
	""" ROI alignment of the feature map.
	
		Removes coarse spatial quantization by bilinear interpolation.

    Parameters
    ----------
    num_classes : output
        Fast R-CNN classification task.

    """
	def __init__(self, pooled_size, spatial_scale=0.03125):
		super(ROIAlignmentLayer, self).__init__()
		self.pooled_size = pooled_size
		self.spatial_scale = spatial_scale
	
	# TODO: maybe the right rois format
	def roi_alignment(self, data, _):
		region = ROIAlign(data=data[0], rois=data[1],
							pooled_size=self.pooled_size, spatial_scale=self.spatial_scale)
		return region, _

	def hybrid_forward(self, F, feature_map, rois):
		feature_map = F.reshape(feature_map, shape=(0,1,512,32,32))
		rois = F.reshape(rois, shape=(0,5,5))
		regions, _ = F.contrib.foreach(self.roi_alignment, [feature_map, rois], [])
		return regions