# import math
import collections
import numpy as np


class Base(object):
    def sample(self, *shape):
        return shape

    def tf(self, img, k=0):
        return img

    def __call__(self, img, dim=3, reuse=False):
        if not reuse:
            im = img if isinstance(img, np.ndarray) else img[0]
            shape = im.shape[1:dim+1]
            self.sample(*shape)

        if isinstance(img, collections.abc.Sequence):
            return [self.tf(x, k) for k, x in enumerate(img)]

        return self.tf(img)

    def __str__(self):
        return 'Identity()'

class Seg_norm(Base):
    def __init__(self, ):
        a = None
        self.seg_table = np.array([0, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62,
               63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121, 122, 161, 162,
               163, 164, 165, 166])
        '''self.seg_table = np.array([0, 1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013,
                 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025,
                 1026, 1027, 1028, 1029, 1030, 1031, 1034, 1035, 2002, 2003, 2005, 2006,
                 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018,
                 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030,
                 2031, 2034, 2035])'''
    def tf(self, img, k=0):
        if k == 0:
            return img
        img_out = np.zeros_like(img)
        for i in range(len(self.seg_table)):
            img_out[img == self.seg_table[i]] = i
        return img_out


class NumpyType(Base):
    def __init__(self, types, num=-1):
        self.types = types # ('float32', 'int64')
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img
        # make this work with both Tensor and Numpy
        return img.astype(self.types[k])

    def __str__(self):
        s = ', '.join([str(s) for s in self.types])
        return 'NumpyType(({}))'.format(s)

