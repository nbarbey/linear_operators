#!/usr/bin/env python
import numpy as np
import scipy
import linear_operators as lo

# Load the infamous Lena image from scipy
im = scipy.lena()
im = im[::4, ::4]
# Generate a convolution model with a 7x7 uniform kernel
model = lo.convolve_ndimage(im.shape, np.ones((7, 7)))
# convolve the original image
data = model * im.ravel()
# add noise to the convolved data
data += 1e0 * np.random.randn(*data.shape)
# define smoothness prior
prior = lo.concatenate([lo.diff(im.shape, axis=i) for i in xrange(im.ndim)])
# generate algorithm
algo = lo.DoubleLoopAlgorithm(model, data, prior,
                              lanczos={"maxiter":100},
                              optimizer=lo.FminNCG,
                              stop_condition=lo.StopCondition(maxiter=5))
# start the estimation algorithm
xe = algo()
# reshape the output as the algorithm only handles vectors
xe.resize(im.shape)
