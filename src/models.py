import numpy as np
from abc import ABCMeta
__all__ = ['LG']


class Model(object):
    __metaclass__ = ABCMeta
    _name = None
    _rates = None
    _freqs = None

    @property
    def name(self):
        return self._name

    @property
    def rates(self):
        return self._rates

    @property
    def freqs(self):
        return self._freqs


lg_rates = np.array(
      [[  0.      ,   0.425093,   0.276818,   0.395144,   2.489084,   0.969894,   1.038545,   2.06604 ,   0.358858,   0.14983 ,   0.395337,   0.536518,   1.124035,   0.253701,   1.177651,   4.727182,   2.139501,   0.180717,   0.218959,   2.54787 ],
       [  0.425093,   0.      ,   0.751878,   0.123954,   0.534551,   2.807908,   0.36397 ,   0.390192,   2.426601,   0.126991,   0.301848,   6.326067,   0.484133,   0.052722,   0.332533,   0.858151,   0.578987,   0.593607,   0.31444 ,   0.170887],
       [  0.276818,   0.751878,   0.      ,   5.076149,   0.528768,   1.695752,   0.541712,   1.437645,   4.509238,   0.191503,   0.068427,   2.145078,   0.371004,   0.089525,   0.161787,   4.008358,   2.000679,   0.045376,   0.612025,   0.083688],
       [  0.395144,   0.123954,   5.076149,   0.      ,   0.062556,   0.523386,   5.24387 ,   0.844926,   0.927114,   0.01069 ,   0.015076,   0.282959,   0.025548,   0.017416,   0.394456,   1.240275,   0.42586 ,   0.02989 ,   0.135107,   0.037967],
       [  2.489084,   0.534551,   0.528768,   0.062556,   0.      ,   0.084808,   0.003499,   0.569265,   0.640543,   0.320627,   0.594007,   0.013266,   0.89368 ,   1.105251,   0.075382,   2.784478,   1.14348 ,   0.670128,   1.165532,   1.959291],
       [  0.969894,   2.807908,   1.695752,   0.523386,   0.084808,   0.      ,   4.128591,   0.267959,   4.813505,   0.072854,   0.582457,   3.234294,   1.672569,   0.035855,   0.624294,   1.223828,   1.080136,   0.236199,   0.257336,   0.210332],
       [  1.038545,   0.36397 ,   0.541712,   5.24387 ,   0.003499,   4.128591,   0.      ,   0.348847,   0.423881,   0.044265,   0.069673,   1.807177,   0.173735,   0.018811,   0.419409,   0.611973,   0.604545,   0.077852,   0.120037,   0.245034],
       [  2.06604 ,   0.390192,   1.437645,   0.844926,   0.569265,   0.267959,   0.348847,   0.      ,   0.311484,   0.008705,   0.044261,   0.296636,   0.139538,   0.089586,   0.196961,   1.73999 ,   0.129836,   0.268491,   0.054679,   0.076701],
       [  0.358858,   2.426601,   4.509238,   0.927114,   0.640543,   4.813505,   0.423881,   0.311484,   0.      ,   0.108882,   0.366317,   0.697264,   0.442472,   0.682139,   0.508851,   0.990012,   0.584262,   0.597054,   5.306834,   0.119013],
       [  0.14983 ,   0.126991,   0.191503,   0.01069 ,   0.320627,   0.072854,   0.044265,   0.008705,   0.108882,   0.      ,   4.145067,   0.159069,   4.273607,   1.112727,   0.078281,   0.064105,   1.033739,   0.11166 ,   0.232523,  10.649107],
       [  0.395337,   0.301848,   0.068427,   0.015076,   0.594007,   0.582457,   0.069673,   0.044261,   0.366317,   4.145067,   0.      ,   0.1375  ,   6.312358,   2.592692,   0.24906 ,   0.182287,   0.302936,   0.619632,   0.299648,   1.702745],
       [  0.536518,   6.326067,   2.145078,   0.282959,   0.013266,   3.234294,   1.807177,   0.296636,   0.697264,   0.159069,   0.1375  ,   0.      ,   0.656604,   0.023918,   0.390322,   0.748683,   1.136863,   0.049906,   0.131932,   0.185202],
       [  1.124035,   0.484133,   0.371004,   0.025548,   0.89368 ,   1.672569,   0.173735,   0.139538,   0.442472,   4.273607,   6.312358,   0.656604,   0.      ,   1.798853,   0.099849,   0.34696 ,   2.020366,   0.696175,   0.481306,   1.898718],
       [  0.253701,   0.052722,   0.089525,   0.017416,   1.105251,   0.035855,   0.018811,   0.089586,   0.682139,   1.112727,   2.592692,   0.023918,   1.798853,   0.      ,   0.094464,   0.361819,   0.165001,   2.457121,   7.803902,   0.654683],
       [  1.177651,   0.332533,   0.161787,   0.394456,   0.075382,   0.624294,   0.419409,   0.196961,   0.508851,   0.078281,   0.24906 ,   0.390322,   0.099849,   0.094464,   0.      ,   1.338132,   0.571468,   0.095131,   0.089613,   0.296501],
       [  4.727182,   0.858151,   4.008358,   1.240275,   2.784478,   1.223828,   0.611973,   1.73999 ,   0.990012,   0.064105,   0.182287,   0.748683,   0.34696 ,   0.361819,   1.338132,   0.      ,   6.472279,   0.248862,   0.400547,   0.098369],
       [  2.139501,   0.578987,   2.000679,   0.42586 ,   1.14348 ,   1.080136,   0.604545,   0.129836,   0.584262,   1.033739,   0.302936,   1.136863,   2.020366,   0.165001,   0.571468,   6.472279,   0.      ,   0.140825,   0.245841,   2.188158],
       [  0.180717,   0.593607,   0.045376,   0.02989 ,   0.670128,   0.236199,   0.077852,   0.268491,   0.597054,   0.11166 ,   0.619632,   0.049906,   0.696175,   2.457121,   0.095131,   0.248862,   0.140825,   0.      ,   3.151815,   0.18951 ],
       [  0.218959,   0.31444 ,   0.612025,   0.135107,   1.165532,   0.257336,   0.120037,   0.054679,   5.306834,   0.232523,   0.299648,   0.131932,   0.481306,   7.803902,   0.089613,   0.400547,   0.245841,   3.151815,   0.      ,   0.249313],
       [  2.54787 ,   0.170887,   0.083688,   0.037967,   1.959291,   0.210332,   0.245034,   0.076701,   0.119013,  10.649107,   1.702745,   0.185202,   1.898718,   0.654683,   0.296501,   0.098369,   2.188158,   0.18951 ,   0.249313,   0.      ]])

lg_freqs = np.array([0.079066, 0.055941, 0.041977, 0.053052, 0.012937, 0.040767, 0.071586, 0.057337, 0.022355, 0.062157, 0.099081, 0.064600, 0.022951, 0.042302, 0.044040, 0.061197, 0.053287, 0.012066, 0.034155, 0.069147])


class LG(Model):
    _name = 'LG'
    _rates = lg_rates
    _freqs = lg_freqs

del np
del ABCMeta
