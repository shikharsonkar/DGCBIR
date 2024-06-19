# -*- coding: utf-8 -*-

from __future__ import print_function

from evaluate import distance, evaluate_class
from DB import Database

from six.moves import cPickle
import numpy as np
import scipy
import itertools
import os
import imageio

# configs for histogram
n_bin   = 12        # histogram bins
n_slice = 3         # slice image
h_type  = 'region'  # global or region
d_type  = 'd1'      # distance type

depth   = 3         # retrieved depth, set to None will count the ap for whole database



# cache dir
cache_dir = 'cache'
if not os.path.exists(cache_dir):
  os.makedirs(cache_dir)


class Color(object):

  def histogram(self, input, n_bin=n_bin, type=h_type, n_slice=n_slice, normalize=True):
  
    if isinstance(input, np.ndarray):  # examinate input type
      img = input.copy()
    else:
      img = imageio.imread(input, mode='RGB')
    height, width, channel = img.shape
    bins = np.linspace(0, 256, n_bin+1, endpoint=True)  # slice bins equally for each channel
  
    if type == 'global':
      hist = self._count_hist(img, n_bin, bins, channel)
  
    elif type == 'region':
      hist = np.zeros((n_slice, n_slice, n_bin ** channel))
      h_silce = np.around(np.linspace(0, height, n_slice+1, endpoint=True)).astype(int)
      w_slice = np.around(np.linspace(0, width, n_slice+1, endpoint=True)).astype(int)
  
      for hs in range(len(h_silce)-1):
        for ws in range(len(w_slice)-1):
          img_r = img[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]  # slice img to regions
          hist[hs][ws] = self._count_hist(img_r, n_bin, bins, channel)
  
    if normalize:
      hist /= np.sum(hist)
  
    return hist.flatten()
  
  
  def _count_hist(self, input, n_bin, bins, channel):
    img = input.copy()
    bins_idx = {key: idx for idx, key in enumerate(itertools.product(np.arange(n_bin), repeat=channel))}  # permutation of bins
    hist = np.zeros(n_bin ** channel)
  
    # cluster every pixels
    for idx in range(len(bins)-1):
      img[(input >= bins[idx]) & (input < bins[idx+1])] = idx
    # add pixels into bins
    height, width, _ = img.shape
    for h in range(height):
      for w in range(width):
        b_idx = bins_idx[tuple(img[h,w])]
        hist[b_idx] += 1
  
    return hist
  
  
  def make_samples(self, db, verbose=True):
    if h_type == 'global':
      sample_cache = "histogram_cache-{}-n_bin{}".format(h_type, n_bin)
    elif h_type == 'region':
      sample_cache = "histogram_cache-{}-n_bin{}-n_slice{}".format(h_type, n_bin, n_slice)
    
    try:
      samples = cPickle.load(open(os.path.join(cache_dir, sample_cache), "rb",  buffering=True))
      if verbose:
        print("Using cache..., config=%s, distance=%s, depth=%s" % (sample_cache, d_type, depth))
    except:
      if verbose:
        print("Counting histogram..., config=%s, distance=%s, depth=%s" % (sample_cache, d_type, depth))
      samples = []
      data = db.get_data()
      for d in data.itertuples():
        d_img, d_cls = getattr(d, "img"), getattr(d, "cls")
        d_hist = self.histogram(d_img, type=h_type, n_bin=n_bin, n_slice=n_slice)
        samples.append({
                        'img':  d_img, 
                        'cls':  d_cls, 
                        'hist': d_hist
                      })
      cPickle.dump(samples, open(os.path.join(cache_dir, sample_cache), "wb",  buffering=True))
    print("length of samples",len(samples))
    return samples


if __name__ == "__main__":
  db = Database()
  data = db.get_data()
  color = Color()

  # test normalize
  hist = color.histogram(data.iloc[0,0], type='global')
  assert hist.sum() - 1 < 1e-9, "normalize false"

  # test histogram bins
  def sigmoid(z):
    a = 1.0 / (1.0 + np.exp(-1. * z))
    return a
  np.random.seed(0)
  IMG = sigmoid(np.random.randn(2,2,3)) * 255
  IMG = IMG.astype(int)
  hist = color.histogram(IMG, type='global', n_bin=4)
  assert np.equal(np.where(hist > 0)[0], np.array([37, 43, 58, 61])).all(), "global histogram implement failed"
  hist = color.histogram(IMG, type='region', n_bin=4, n_slice=2)
  assert np.equal(np.where(hist > 0)[0], np.array([58, 125, 165, 235])).all(), "region histogram implement failed"

  # examinate distance
  np.random.seed(1)
  IMG = sigmoid(np.random.randn(4,4,3)) * 255
  IMG = IMG.astype(int)
  hist = color.histogram(IMG, type='region', n_bin=4, n_slice=2)
  IMG2 = sigmoid(np.random.randn(4,4,3)) * 255
  IMG2 = IMG2.astype(int)
  hist2 = color.histogram(IMG2, type='region', n_bin=4, n_slice=2)
  assert distance(hist, hist2, d_type='d1') == 2, "d1 implement failed"
  assert distance(hist, hist2, d_type='d2-norm') == 2, "d2 implement failed"
db = Database()
  # evaluate database
 # evaluate database
APs = evaluate_class(db, f_class=Color, d_type=d_type, depth=depth)
cls_MAPs = []
for cls, cls_APs in APs.items():
    MAP = np.mean(cls_APs)
    if not np.isnan(MAP):  # Check if MAP is not nan
        print("Class {}, MAP {}".format(cls, MAP))
        cls_MAPs.append(MAP)
    else:
        print("Class {}, No relevant images found.".format(cls))

if cls_MAPs:  # Check if the list is not empty
    print("MMAP", np.nanmean(cls_MAPs))
else:
    print("No relevant images found for any class.")