from mxnet import nd, gluon
from mxnet.gluon import nn

class Residual(gluon.nn.HybridBlock):
	def __init__(self, num_channels, use_1x1conv=False, strides=1):
		super(Residual, self).__init__()

		self.conv1 = nn.Conv2D(num_channels, kernel_size=3, strides=strides, padding=1)
		self.bn1 = nn.BatchNorm()
		self.relu1 = nn.Activation('relu')
		self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
		self.bn2 = nn.BatchNorm()

		if use_1x1conv:
			self.conv3 = nn.Conv2D(num_channels, kernel_size=1, strides=strides)
		else:
			self.conv3 = None
		
		self.relu2 = nn.Activation('relu')

	def hybrid_forward(self, F, X):
		Y = self.relu1(self.bn1(self.conv1(X)))
		y = self.bn2(self.conv2(Y))

		if self.conv3:
			x = self.conv3(X)
		else:
			x = X
		return self.relu2(y + x)

def stack_resnet_block(num_residuals, num_channels, first_block=False):
	out = nn.HybridSequential()
	for i in range(num_residuals):
		if i==0 and not first_block:
			out.add(Residual(num_channels, use_1x1conv=True, strides=2))
		else:
			out.add(Residual(num_channels))
	return out


class RoentgenResnet(gluon.nn.HybridBlock):
	"""Feature extraction using standard resnet architecture.

    Parameters
    ----------
    conv_arch : number of residuals and channels
        with 1x1conv at each of the first blocks and increasing feature channels.

    """
	def __init__(self,num_channels, conv_arch=[(2, 64), (2, 128), (2, 256), (2, 512)]):
		super(RoentgenResnet, self).__init__()

		self.conv0 = nn.Conv2D(num_channels, kernel_size=7, strides=2, padding=3)
		self.bn0 = nn.BatchNorm()
		self.relu0 = nn.Activation('relu')
		self.pool0 = nn.MaxPool2D(pool_size=3, strides=2, padding=1)

		self.residuals = nn.HybridSequential()

		for i, (num_residuals, num_channels) in enumerate(conv_arch):
			if i==0:
				self.residuals.add(stack_resnet_block(num_residuals, num_channels, first_block=True))
			else:
				self.residuals.add(stack_resnet_block(num_residuals, num_channels))
	
	def hybrid_forward(self, F, X):
		Y = self.relu0(self.bn0(self.conv0(X)))
		x = self.residuals(self.pool0(Y))
		return x