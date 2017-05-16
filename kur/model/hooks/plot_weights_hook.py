################################
# prepare examine tools
from pdb import set_trace
from pprint import pprint
from inspect import getdoc, getmembers, getsourcelines, getmodule, getfullargspec, getargvalues
"""
Copyright 2017 Deepgram

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import colorsys
import os

import itertools
from collections import OrderedDict

import numpy
import tempfile

from . import TrainingHook
from ...loggers import PersistentLogger, Statistic

import logging
import matplotlib.pyplot as plt
import numpy as np
import math
logger = logging.getLogger(__name__)
from ...utils import DisableLogging, idx

###############################################################################
class PlotWeightsHook(TrainingHook):
	""" Hook for creating plots of loss.
	"""

	###########################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the hook.
		"""
		return 'plot_weights'

	###########################################################################
	def __init__(self, layer_names=None, plot_directory=None, weight_file=None, with_weights=None, plot_every_n_epochs=None, animate_layers=None, *args, **kwargs):
		""" Creates a new plot hook for plotting weights of layers
		"""

		super().__init__(*args, **kwargs)

		# added self.layer_names
		self.layer_names = layer_names
		self.directory = plot_directory
		if not os.path.exists(self.directory):
			os.makedirs(self.directory)

		self.plot_every_n_epochs = plot_every_n_epochs

		if weight_file is None:
			self.weight_file = None
		else:
			self.weight_file = weight_file

		self.with_weights = with_weights

		try:
			import matplotlib					# pylint: disable=import-error
		except:
			logger.exception('Failed to import "matplotlib". Make sure it is '
				'installed, and if you have continued trouble, please check '
				'out our troubleshooting page: https://kur.deepgram.com/'
				'troubleshooting.html#plotting')
			raise

		# Set the matplotlib backend to one of the known backends.
		matplotlib.use('Agg')

	###########################################################################
	def notify(self, status, log=None, info=None, model=None):
		""" Creates the plot.
		"""

		from matplotlib import pyplot as plt	# pylint: disable=import-error
		if status not in (
			# the plotting is allowed only at end of epoch
			TrainingHook.EPOCH_END,
		):

			return

		weight_path = None
		tempdir = tempfile.mkdtemp()
		weight_path = os.path.join(tempdir, 'current_epoch_model')
		model.save(weight_path)

		# only plot conv layers (lots of squared images)
		def plot_conv_layer(layer_out, layer_name):

			values = layer_out
		    # In my experiment example, this layer dim (1, 31, 31, 64)
			num_filters = values.shape[3]

			num_grids = math.ceil(math.sqrt(num_filters))

			fig, axes = plt.subplots(num_grids, num_grids)
			fig.suptitle(layer_name+", shape:"+str(values.shape), fontsize="x-large")
			# global title
			# fig.suptitle(layer_name+", shape:"+str(values.shape), fontsize="x-large")

			for i, ax in enumerate(axes.flat):
				if i<num_filters:

					img = values[0, :, :, i]
					ax.imshow(img, interpolation='nearest', cmap='binary') # binary, seismic


		        # # Remove ticks from the plot.
				ax.set_xticks([])
				ax.set_yticks([])

			fig.savefig('{}/{}_epoch_{}.png'.format(self.directory, layer_name, info['epoch']))
			# so that previous plots won't affect later plot
			plt.clf()

		# plot layers with 2-d or 1-d, what about layers with uneven 3-d
		def plot_uneven_layer(layer_out, layer_name):

			w = layer_out

			w_plot = None
			if len(w.shape)==2 and max(w.shape)>20 :
				s1, s2 = w.shape
				if s1 < s2:
					w_plot = w[:, :20]
				else:
					w_plot = w[:20, :]
			elif len(w.shape)==2 and max(w.shape)<=20:
				w_plot = w

			elif len(w.shape) == 1 and w.shape[0]>20:
				# reshape to 2-d, as plt.imshow only work on 2-d
				w_plot = w.reshape(1,-1)[1,:20]

			elif len(w.shape) == 1 and w.shape[0]<=20:
				w_plot = w.reshape(1,-1)

			elif len(w.shape) > 2:
				logger.error("w.shape length >=3, need extra code to handle it")

			# set_trace()
			plt.imshow(w_plot, interpolation='nearest', cmap='bone', origin='lower')

			# set the size of color bar
			plt.colorbar(shrink=.72)
			plt.suptitle("{}_layer, shape: {}".format(layer_name, w.shape), fontsize="x-large")
			plt.xticks(())
			plt.yticks(())


			plt.savefig('{}/{}_epoch_{}.png'.format(self.directory, layer_name, info['epoch']))
			# so that previous plots won't affect later plot
			plt.clf()

			   							# keywords of weights
			# only plot 1-d, 2-d, what about 3-d uneven weights
		def plot_uneven_weights(kernel_filename, weights_keywords):

			w = idx.load(kernel_filename)
			if len(w.shape)==2 and max(w.shape)>20 :
				s1, s2 = w.shape
				if s1 < s2:
					w_plot = w[:, :20]
				else:
					w_plot = w[:20, :]

			elif len(w.shape) == 1 and w.shape[0]>20:
				w_plot = w[:20]
			else:
				logger.warning("w.shape length is not 1 nor 2")

			plt.imshow(w_plot, interpolation='nearest', cmap='bone', origin='lower')

			# set the size of color bar
			plt.colorbar(shrink=.72)
			plt.suptitle("{}_weights, 1st 20 out of shape: {}".format(weights_keywords, w.shape), fontsize="x-large")
			plt.xticks(())
			plt.yticks(())


			filename_cut_dir = kernel_filename[kernel_filename.find(weights_keywords) :]

			plt.savefig('{}/{}_epoch_{}.png'.format(self.directory, filename_cut_dir, info['epoch']))
			# so that previous plots won't affect later plot
			plt.clf()

		def plot_conv_weights(kernel_filename, input_channel=0):
			w = idx.load(kernel_filename)

			w_min = np.min(w)
			w_max = np.max(w)

			s1, s2, s3, s4 = w.shape
			if s1 > s4:
				w = w.reshape((s3, s4, s2, s1))

			num_filters = w.shape[3]
			num_grids = math.ceil(math.sqrt(num_filters))

			fig, axes = plt.subplots(num_grids, num_grids)
			fig.suptitle("Conv_weights, shape:"+str(w.shape), fontsize="x-large")
			for i, ax in enumerate(axes.flat):
				if i<num_filters:

					img = w[:, :, input_channel, i]
					ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')

				# if i == 0:
				# 	ax.set_title("loss: {}\nshape ({}), conv_weight".format(round(info['Validation loss'][None]['labels'], 3), w.shape))

				ax.set_xticks([])
				ax.set_yticks([])

			filename_cut_dir = kernel_filename[kernel_filename.find("convol") :]

			fig.savefig('{}/{}_epoch_{}.png'.format(self.directory, filename_cut_dir, info['epoch']))
			# so that previous plots won't affect later plot
			plt.clf()


		if info['epoch'] == 1 or info['epoch'] % self.plot_every_n_epochs == 0:

			valid_weights_filenames = []

			if self.weight_file is None:
				self.weight_file = weight_path

			for dirpath, _, filenames in os.walk(self.weight_file):

				for this_file in filenames:
					valid_weights_filenames.append(dirpath+"/"+this_file)

			for this_file in valid_weights_filenames:
				for weight_keywords in self.with_weights:

					if this_file.find(weight_keywords[0]) > -1 and this_file.find(weight_keywords[1]) > -1:

						if weight_keywords[0].find("recurrent") > -1 or weight_keywords[1].find("recurrent") > -1:
							weight_keywords = "recurrent"
							plot_uneven_weights(this_file, weight_keywords)

						if weight_keywords[0].find("convol") > -1 or weight_keywords[1].find("convol") > -1:
							plot_conv_weights(this_file)

						if weight_keywords[0].find("dense") > -1 or weight_keywords[1].find("dense") > -1:
							weight_keywords = "dense"
							plot_uneven_weights(this_file, weight_keywords)

			# if layer_names are not given, then don't plot layers
			if self.layer_names is None:
				return

			for layer_name in self.layer_names:
				# take only one sample to plot
				layer_out = info['inter_layers_outputs'][layer_name]

				# to plot 3-d or more dim layers
				if len(layer_out.shape) > 2:

					output = layer_out[0]

					# make sure img_dim is (1, w, h, num_filters), here (1, 31, 31, 64)
					# plot 3 supposedly color channel of input image
					img_dim = (1,) + output.shape

					output_reshape = output.reshape(img_dim)

					plot_conv_layer(output_reshape, layer_name)

				# plot only 2 dim layer or vector layer
				else:

					output = layer_out[0]
					plot_uneven_layer(output, layer_name)
