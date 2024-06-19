# -*- coding: utf-8 -*-

from __future__ import print_function

from evaluate import evaluate_class
from DB import Database

from skimage.feature import hog
from skimage import color

import imageio
from six.moves import cPickle
import numpy as np
import scipy
import os

n_bin    = 10
n_slice  = 6
n_orient = 8
p_p_c    = (2, 2)
c_p_b    = (1, 1)
h_type   = 'region'
d_type   = 'd1'

depth    = 5



# cache dir
cache_dir = 'cache'
if not os.path.exists(cache_dir):
  os.makedirs(cache_dir)


class HOG(object):

  def histogram(self, input, n_bin=n_bin, type=h_type, n_slice=n_slice, normalize=True):

    if isinstance(input, np.ndarray):  # examinate input type
      img = input.copy()
    else:
      img = imageio.imread(input, mode='RGB')
    height, width, channel = img.shape
  
    if type == 'global':
      hist = self._HOG(img, n_bin)
  
    elif type == 'region':
      hist = np.zeros((n_slice, n_slice, n_bin))
      h_silce = np.around(np.linspace(0, height, n_slice+1, endpoint=True)).astype(int)
      w_slice = np.around(np.linspace(0, width, n_slice+1, endpoint=True)).astype(int)
  
      for hs in range(len(h_silce)-1):
        for ws in range(len(w_slice)-1):
          img_r = img[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]  # slice img to regions
          hist[hs][ws] = self._HOG(img_r, n_bin)
  
    if normalize:
      hist /= np.sum(hist)
  
    return hist.flatten()

  def _HOG(self, img, n_bin, normalize=True):
    image = color.rgb2gray(img)
    fd = hog(image, orientations=n_orient, pixels_per_cell=p_p_c, cells_per_block=c_p_b)
    bins = np.linspace(0, np.max(fd), n_bin+1, endpoint=True)
    hist, _ = np.histogram(fd, bins=bins)
  
    if normalize:
      hist = np.array(hist) / np.sum(hist)
  
    return hist

  def make_samples(self, db, verbose=True):
    if h_type == 'global':
      sample_cache = "HOG-{}-n_bin{}-n_orient{}-ppc{}-cpb{}".format(h_type, n_bin, n_orient, p_p_c, c_p_b)
    elif h_type == 'region':
      sample_cache = "HOG-{}-n_bin{}-n_slice{}-n_orient{}-ppc{}-cpb{}".format(h_type, n_bin, n_slice, n_orient, p_p_c, c_p_b)
  
    try:
      samples = cPickle.load(open(os.path.join(cache_dir, sample_cache), "rb", True))
      for sample in samples:
        sample['hist'] /= np.sum(sample['hist'])  # normalize
      if verbose:
        print("Using cache..., config=%s, distance=%s, depth=%s" % (sample_cache, d_type, depth))
    except:
      if verbose:
        print("Counting histogram..., config=%s, distance=%s, depth=%s" % (sample_cache, d_type, depth))

      samples = []
      data = db.get_data()
      for d in data.itertuples():
        d_img, d_cls = getattr(d, "img"), getattr(d, "cls")
        d_hist = self.histogram(d_img, type=h_type, n_slice=n_slice)
        samples.append({
                        'img':  d_img, 
                        'cls':  d_cls, 
                        'hist': d_hist
                      })
      cPickle.dump(samples, open(os.path.join(cache_dir, sample_cache), "wb", True))

    return samples


if __name__ == "__main__":
  db = Database()

  # evaluate database
  APs = evaluate_class(db, f_class=HOG, d_type=d_type, depth=depth)
  cls_MAPs = []
  for cls, cls_APs in APs.items():
    MAP = np.mean(cls_APs)
    print("Class {}, MAP {}".format(cls, MAP))
    cls_MAPs.append(MAP)
  print("MMAP", np.mean(cls_MAPs))
