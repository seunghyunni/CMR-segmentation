import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable
from partialconv2d import PartialConv2d
import torchvision.utils as v_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms

class Modified2DUNet(nn.Module):
	def __init__(self, in_channels, n_classes, last_activation, base_n_filter = 16):
		super(Modified2DUNet, self).__init__()
		self.in_channels = in_channels
		self.n_classes = n_classes
		self.base_n_filter = base_n_filter

		self.lrelu = nn.LeakyReLU()
		self.dropout2d = nn.Dropout2d(p=0.6)
		
		self.last_activation = last_activation

		# Level 1 context pathway
		self.conv2d_c1_1 = PartialConv2d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
		self.conv2d_c1_2 = PartialConv2d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
		self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
		self.inorm2d_c1 = nn.InstanceNorm2d(self.base_n_filter)

		# Level 2 context pathway
		self.conv2d_c2 = PartialConv2d(self.base_n_filter, self.base_n_filter*2, kernel_size=3, stride=2, padding=1, bias=False)
		self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter*2, self.base_n_filter*2)
		self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter*2, self.base_n_filter*2)
		self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter*2, self.base_n_filter*2)
		self.inorm2d_c2 = nn.InstanceNorm2d(self.base_n_filter*2)

		# Level 3 context pathway
		self.conv2d_c3 = PartialConv2d(self.base_n_filter*2, self.base_n_filter*4, kernel_size=3, stride=2, padding=1, bias=False)
		self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter*4, self.base_n_filter*4)
		self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter*4, self.base_n_filter*4)
		self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter*4, self.base_n_filter*4)
		self.inorm2d_c3 = nn.InstanceNorm2d(self.base_n_filter*4)

		# Level 4 context pathway
		self.conv2d_c4 = PartialConv2d(self.base_n_filter*4, self.base_n_filter*8, kernel_size=3, stride=2, padding=1, bias=False)
		self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter*8, self.base_n_filter*8)
		self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter*8, self.base_n_filter*8)
		self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter*8, self.base_n_filter*8)
		self.inorm2d_c4 = nn.InstanceNorm2d(self.base_n_filter*8)

		# Level 5 context pathway, level 0 localization pathway
		self.conv2d_c5 = PartialConv2d(self.base_n_filter*8, self.base_n_filter*16, kernel_size=3, stride=2, padding=1, bias=False)
		self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter*16, self.base_n_filter*16)
		self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter*16, self.base_n_filter*16)
		self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter*16, self.base_n_filter*16)
		self.norm_lrelu_upscale_conv_norm_lrelu_l0_1 = self.norm_lrelu_upscale_conv_norm_lrelu_1(self.base_n_filter*16)
		self.norm_lrelu_upscale_conv_norm_lrelu_l0_2 = self.norm_lrelu_upscale_conv_norm_lrelu_2(self.base_n_filter*16, self.base_n_filter*8)

		self.conv2d_l0 = PartialConv2d(self.base_n_filter*8, self.base_n_filter*8, kernel_size = 1, stride=1, padding=0, bias=False)
		self.inorm2d_l0 = nn.InstanceNorm2d(self.base_n_filter*8)

		# Level 1 localization pathway
		self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter*16, self.base_n_filter*16)
		self.conv2d_l1 = PartialConv2d(self.base_n_filter*16, self.base_n_filter*8, kernel_size=1, stride=1, padding=0, bias=False)
		self.norm_lrelu_upscale_conv_norm_lrelu_l1_1 = self.norm_lrelu_upscale_conv_norm_lrelu_1(self.base_n_filter*8)
		self.norm_lrelu_upscale_conv_norm_lrelu_l1_2 = self.norm_lrelu_upscale_conv_norm_lrelu_2(self.base_n_filter*8, self.base_n_filter*4)

		# Level 2 localization pathway
		self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter*8, self.base_n_filter*8)
		self.conv2d_l2 = PartialConv2d(self.base_n_filter*8, self.base_n_filter*4, kernel_size=1, stride=1, padding=0, bias=False)
		self.norm_lrelu_upscale_conv_norm_lrelu_l2_1 = self.norm_lrelu_upscale_conv_norm_lrelu_1(self.base_n_filter*4)
		self.norm_lrelu_upscale_conv_norm_lrelu_l2_2 = self.norm_lrelu_upscale_conv_norm_lrelu_2(self.base_n_filter*4, self.base_n_filter*2)

		# Level 3 localization pathway
		self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter*4, self.base_n_filter*4)
		self.conv2d_l3 = PartialConv2d(self.base_n_filter*4, self.base_n_filter*2, kernel_size=1, stride=1, padding=0, bias=False)
		self.norm_lrelu_upscale_conv_norm_lrelu_l3_1 = self.norm_lrelu_upscale_conv_norm_lrelu_1(self.base_n_filter*2)
		self.norm_lrelu_upscale_conv_norm_lrelu_l3_2 = self.norm_lrelu_upscale_conv_norm_lrelu_2(self.base_n_filter*2, self.base_n_filter)

		# Level 4 localization pathway
		self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter*2, self.base_n_filter*2)
		self.conv2d_l4 = PartialConv2d(self.base_n_filter*2, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)

		self.ds2_1x1_conv2d = PartialConv2d(self.base_n_filter*8, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)
		self.ds3_1x1_conv2d = PartialConv2d(self.base_n_filter*4, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)




	def conv_norm_lrelu(self, feat_in, feat_out):
		return nn.Sequential(
			PartialConv2d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
			nn.InstanceNorm2d(feat_out),
			nn.LeakyReLU())

	def norm_lrelu_conv(self, feat_in, feat_out):
		return nn.Sequential(
			nn.InstanceNorm2d(feat_in),
			nn.LeakyReLU(),
			PartialConv2d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

	def lrelu_conv(self, feat_in, feat_out):
		return nn.Sequential(
			nn.LeakyReLU(),
			PartialConv2d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

	def norm_lrelu_upscale_conv_norm_lrelu_1(self, feat_in):
		return nn.Sequential(
			nn.InstanceNorm2d(feat_in),
			nn.LeakyReLU())

	def norm_lrelu_upscale_conv_norm_lrelu_2(self, feat_in, feat_out):
		return nn.Sequential(
			# should be feat_in*2 or feat_in
			PartialConv2d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
			nn.InstanceNorm2d(feat_out),
			nn.LeakyReLU())

	def forward(self, x):
		#  Level 1 context pathway
		out = self.conv2d_c1_1(x)
		residual_1 = out
		out = self.lrelu(out)
		out = self.conv2d_c1_2(out)
		out = self.dropout2d(out)
		out = self.lrelu_conv_c1(out)
		# Element Wise Summation
		out += residual_1
		context_1 = self.lrelu(out)
		out = self.inorm2d_c1(out)
		out = self.lrelu(out)

		# Level 2 context pathway
		out = self.conv2d_c2(out)
		residual_2 = out
		out = self.norm_lrelu_conv_c2(out)
		out = self.norm_lrelu_conv_c2(out)
		out = self.dropout2d(out)
		out = self.norm_lrelu_conv_c2(out)
		out = self.norm_lrelu_conv_c2(out)
		out += residual_2
		out = self.inorm2d_c2(out)
		out = self.lrelu(out)
		context_2 = out

		# Level 3 context pathway
		out = self.conv2d_c3(out)
		residual_3 = out
		out = self.norm_lrelu_conv_c3(out)
		out = self.norm_lrelu_conv_c3(out)
		out = self.dropout2d(out)
		out = self.norm_lrelu_conv_c3(out)
		out = self.norm_lrelu_conv_c3(out)
		out += residual_3
		out = self.inorm2d_c3(out)
		out = self.lrelu(out)
		context_3 = out

		# Level 4 context pathway
		out = self.conv2d_c4(out)
		residual_4 = out
		out = self.norm_lrelu_conv_c4(out)
		out = self.norm_lrelu_conv_c4(out)
		out = self.dropout2d(out)
		out = self.norm_lrelu_conv_c4(out)
		out = self.norm_lrelu_conv_c4(out)
		out += residual_4
		out = self.inorm2d_c4(out)
		out = self.lrelu(out)
		context_4 = out

		# Level 5
		out = self.conv2d_c5(out)
		residual_5 = out
		out = self.norm_lrelu_conv_c5(out)
		out = self.norm_lrelu_conv_c5(out)
		out = self.dropout2d(out)
		out = self.norm_lrelu_conv_c5(out)
		out = self.norm_lrelu_conv_c5(out)
		out += residual_5
		out = self.norm_lrelu_upscale_conv_norm_lrelu_l0_1(out)
		out = F.interpolate(out, scale_factor=2, mode='nearest')
		out = self.norm_lrelu_upscale_conv_norm_lrelu_l0_2(out)

		out = self.conv2d_l0(out)
		out = self.inorm2d_l0(out)
		out = self.lrelu(out)

		# Level 1 localization pathway
		out = F.interpolate(out, size = context_4.size()[-2:])
		out = torch.cat([out, context_4], dim=1)
		out = self.conv_norm_lrelu_l1(out)
		out = self.conv2d_l1(out)
		out = self.norm_lrelu_upscale_conv_norm_lrelu_l1_1(out)
		out = F.interpolate(out, scale_factor=2, mode='nearest')
		out = self.norm_lrelu_upscale_conv_norm_lrelu_l1_2(out)


		# Level 2 localization pathway
		out = F.interpolate(out, size = context_3.size()[-2:])
		out = torch.cat([out, context_3], dim=1)
		out = self.conv_norm_lrelu_l2(out)
		ds2 = out
		out = self.conv2d_l2(out)
		out = self.norm_lrelu_upscale_conv_norm_lrelu_l2_1(out)
		out = F.interpolate(out, scale_factor=2, mode='nearest')
		out = self.norm_lrelu_upscale_conv_norm_lrelu_l2_2(out)

		# Level 3 localization pathway
		out = F.interpolate(out, size = context_2.size()[-2:])
		out = torch.cat([out, context_2], dim=1)
		out = self.conv_norm_lrelu_l3(out)
		ds3 = out
		out = self.conv2d_l3(out)
		out = self.norm_lrelu_upscale_conv_norm_lrelu_l3_1(out)
		out = F.interpolate(out, scale_factor=2, mode='nearest')
		out = self.norm_lrelu_upscale_conv_norm_lrelu_l3_2(out)

		# Level 4 localization pathway
		out = F.interpolate(out, size = context_1.size()[-2:])
		out = torch.cat([out, context_1], dim=1)
		out = self.conv_norm_lrelu_l4(out)
		out_pred = self.conv2d_l4(out)

		ds2_1x1_conv = self.ds2_1x1_conv2d(ds2)
		ds1_ds2_sum_upscale = F.interpolate(ds2_1x1_conv, scale_factor=2, mode='nearest')
		ds3_1x1_conv = self.ds3_1x1_conv2d(ds3)
		ds1_ds2_sum_upscale = F.interpolate(ds1_ds2_sum_upscale, size = ds3_1x1_conv.size()[-2:])
		ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
		ds1_ds2_sum_upscale_ds3_sum_upscale = F.interpolate(ds1_ds2_sum_upscale_ds3_sum, scale_factor=2, mode='nearest')

		out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale
		# out = out.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.n_classes)
		#out = out.view(-1, self.n_classes)
		out2 = nn.Sigmoid()(out)
		res = [out, out2]
		return res
