#!/usr/bin/env python
import numpy as np
import scipy
import linear_operators as lo

# Load the infamous Lena image from scipy
im = scipy.lena()
# Generate a convolution model with a 3x3 uniform kernel
kernel = np.ones((5, 5))
model = lo.convolve_ndimage(im.shape, kernel)
# convolve the original image
data = model(im)
# add noise to the convolved data
noise = 1e1 * np.random.randn(*data.shape)
data += noise
# define smoothness priors
prior = [lo.diff(im.shape, axis=i) for i in xrange(im.ndim)]
hypers = (1e1, 1e1)
# generate an conjugate gradient algorithm from model, data and priors
algo = lo.QuadraticConjugateGradient(model, data, prior, hypers,
                                     stop_condition=lo.StopCondition(gtol=1e-5))
# start the estimation algorithm
xe = algo()
# reshape the output as the algorithm only handles vectors
xe.resize(im.shape)
