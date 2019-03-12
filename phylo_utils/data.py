import numpy as np

# Protein model states: 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'
lg_rates = np.ascontiguousarray(
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
lg_freqs = np.ascontiguousarray([0.079066, 0.055941, 0.041977, 0.053052, 0.012937, 0.040767, 0.071586, 0.057337, 0.022355, 0.062157, 0.099081, 0.064600, 0.022951, 0.042302, 0.044040, 0.061197, 0.053287, 0.012066, 0.034155, 0.069147])
lg_freqs.setflags(write=False)
lg_rates.setflags(write=False)

wag_rates = np.ascontiguousarray(
      [[ 0.       ,  0.551571 ,  0.509848 ,  0.738998 ,  1.02704  ,  0.908598 ,  1.58285  ,  1.41672  ,  0.316954 ,  0.193335 ,  0.397915 ,  0.906265 ,  0.893496 ,  0.210494 ,  1.43855  ,  3.37079  ,  2.12111  ,  0.113133 ,  0.240735 ,  2.00601  ],
       [ 0.551571 ,  0.       ,  0.635346 ,  0.147304 ,  0.528191 ,  3.0355   ,  0.439157 ,  0.584665 ,  2.13715  ,  0.186979 ,  0.497671 ,  5.35142  ,  0.683162 ,  0.102711 ,  0.679489 ,  1.22419  ,  0.554413 ,  1.16392  ,  0.381533 ,  0.251849 ],
       [ 0.509848 ,  0.635346 ,  0.       ,  5.42942  ,  0.265256 ,  1.54364  ,  0.947198 ,  1.12556  ,  3.95629  ,  0.554236 ,  0.131528 ,  3.01201  ,  0.198221 ,  0.0961621,  0.195081 ,  3.97423  ,  2.03006  ,  0.0719167,  1.086    ,  0.196246 ],
       [ 0.738998 ,  0.147304 ,  5.42942  ,  0.       ,  0.0302949,  0.616783 ,  6.17416  ,  0.865584 ,  0.930676 ,  0.039437 ,  0.0848047,  0.479855 ,  0.103754 ,  0.0467304,  0.423984 ,  1.07176  ,  0.374866 ,  0.129767 ,  0.325711 ,  0.152335 ],
       [ 1.02704  ,  0.528191 ,  0.265256 ,  0.0302949,  0.       ,  0.0988179,  0.021352 ,  0.306674 ,  0.248972 ,  0.170135 ,  0.384287 ,  0.0740339,  0.390482 ,  0.39802  ,  0.109404 ,  1.40766  ,  0.512984 ,  0.71707  ,  0.543833 ,  1.00214  ],
       [ 0.908598 ,  3.0355   ,  1.54364  ,  0.616783 ,  0.0988179,  0.       ,  5.46947  ,  0.330052 ,  4.29411  ,  0.113917 ,  0.869489 ,  3.8949   ,  1.54526  ,  0.0999208,  0.933372 ,  1.02887  ,  0.857928 ,  0.215737 ,  0.22771  ,  0.301281 ],
       [ 1.58285  ,  0.439157 ,  0.947198 ,  6.17416  ,  0.021352 ,  5.46947  ,  0.       ,  0.567717 ,  0.570025 ,  0.127395 ,  0.154263 ,  2.58443  ,  0.315124 ,  0.0811339,  0.682355 ,  0.704939 ,  0.822765 ,  0.156557 ,  0.196303 ,  0.588731 ],
       [ 1.41672  ,  0.584665 ,  1.12556  ,  0.865584 ,  0.306674 ,  0.330052 ,  0.567717 ,  0.       ,  0.24941  ,  0.0304501,  0.0613037,  0.373558 ,  0.1741   ,  0.049931 ,  0.24357  ,  1.34182  ,  0.225833 ,  0.336983 ,  0.103604 ,  0.187247 ],
       [ 0.316954 ,  2.13715  ,  3.95629  ,  0.930676 ,  0.248972 ,  4.29411  ,  0.570025 ,  0.24941  ,  0.       ,  0.13819  ,  0.499462 ,  0.890432 ,  0.404141 ,  0.679371 ,  0.696198 ,  0.740169 ,  0.473307 ,  0.262569 ,  3.87344  ,  0.118358 ],
       [ 0.193335 ,  0.186979 ,  0.554236 ,  0.039437 ,  0.170135 ,  0.113917 ,  0.127395 ,  0.0304501,  0.13819  ,  0.       ,  3.17097  ,  0.323832 ,  4.25746  ,  1.05947  ,  0.0999288,  0.31944  ,  1.45816  ,  0.212483 ,  0.42017  ,  7.8213   ],
       [ 0.397915 ,  0.497671 ,  0.131528 ,  0.0848047,  0.384287 ,  0.869489 ,  0.154263 ,  0.0613037,  0.499462 ,  3.17097  ,  0.       ,  0.257555 ,  4.85402  ,  2.11517  ,  0.415844 ,  0.344739 ,  0.326622 ,  0.665309 ,  0.398618 ,  1.80034  ],
       [ 0.906265 ,  5.35142  ,  3.01201  ,  0.479855 ,  0.0740339,  3.8949   ,  2.58443  ,  0.373558 ,  0.890432 ,  0.323832 ,  0.257555 ,  0.       ,  0.934276 ,  0.088836 ,  0.556896 ,  0.96713  ,  1.38698  ,  0.137505 ,  0.133264 ,  0.305434 ],
       [ 0.893496 ,  0.683162 ,  0.198221 ,  0.103754 ,  0.390482 ,  1.54526  ,  0.315124 ,  0.1741   ,  0.404141 ,  4.25746  ,  4.85402  ,  0.934276 ,  0.       ,  1.19063  ,  0.171329 ,  0.493905 ,  1.51612  ,  0.515706 ,  0.428437 ,  2.05845  ],
       [ 0.210494 ,  0.102711 ,  0.0961621,  0.0467304,  0.39802  ,  0.0999208,  0.0811339,  0.049931 ,  0.679371 ,  1.05947  ,  2.11517  ,  0.088836 ,  1.19063  ,  0.       ,  0.161444 ,  0.545931 ,  0.171903 ,  1.52964  ,  6.45428  ,  0.649892 ],
       [ 1.43855  ,  0.679489 ,  0.195081 ,  0.423984 ,  0.109404 ,  0.933372 ,  0.682355 ,  0.24357  ,  0.696198 ,  0.0999288,  0.415844 ,  0.556896 ,  0.171329 ,  0.161444 ,  0.       ,  1.61328  ,  0.795384 ,  0.139405 ,  0.216046 ,  0.314887 ],
       [ 3.37079  ,  1.22419  ,  3.97423  ,  1.07176  ,  1.40766  ,  1.02887  ,  0.704939 ,  1.34182  ,  0.740169 ,  0.31944  ,  0.344739 ,  0.96713  ,  0.493905 ,  0.545931 ,  1.61328  ,  0.       ,  4.37802  ,  0.523742 ,  0.786993 ,  0.232739 ],
       [ 2.12111  ,  0.554413 ,  2.03006  ,  0.374866 ,  0.512984 ,  0.857928 ,  0.822765 ,  0.225833 ,  0.473307 ,  1.45816  ,  0.326622 ,  1.38698  ,  1.51612  ,  0.171903 ,  0.795384 ,  4.37802  ,  0.       ,  0.110864 ,  0.291148 ,  1.38823  ],
       [ 0.113133 ,  1.16392  ,  0.0719167,  0.129767 ,  0.71707  ,  0.215737 ,  0.156557 ,  0.336983 ,  0.262569 ,  0.212483 ,  0.665309 ,  0.137505 ,  0.515706 ,  1.52964  ,  0.139405 ,  0.523742 ,  0.110864 ,  0.       ,  2.48539  ,  0.365369 ],
       [ 0.240735 ,  0.381533 ,  1.086    ,  0.325711 ,  0.543833 ,  0.22771  ,  0.196303 ,  0.103604 ,  3.87344  ,  0.42017  ,  0.398618 ,  0.133264 ,  0.428437 ,  6.45428  ,  0.216046 ,  0.786993 ,  0.291148 ,  2.48539  ,  0.       ,  0.31473  ],
       [ 2.00601  ,  0.251849 ,  0.196246 ,  0.152335 ,  1.00214  ,  0.301281 ,  0.588731 ,  0.187247 ,  0.118358 ,  7.8213   ,  1.80034  ,  0.305434 ,  2.05845  ,  0.649892 ,  0.314887 ,  0.232739 ,  1.38823  ,  0.365369 ,  0.31473  ,  0.       ]])
wag_freqs = np.ascontiguousarray([0.0866279, 0.043972, 0.0390894, 0.0570451, 0.0193078, 0.0367281, 0.0580589, 0.0832518, 0.0244313, 0.048466, 0.086209, 0.0620286, 0.0195027, 0.0384319, 0.0457631, 0.0695179, 0.0610127, 0.0143859, 0.0352742, 0.0708956])
wag_freqs.setflags(write=False)
wag_rates.setflags(write=False)

fixed_equal_nucleotide_rates = np.ascontiguousarray(
        [[0.0, 1.0, 1.0, 1.0],
         [1.0, 0.0, 1.0, 1.0],
         [1.0, 1.0, 0.0, 1.0],
         [1.0, 1.0, 1.0, 0.0]])
fixed_equal_nucleotide_rates.setflags(write=False)
fixed_equal_nucleotide_frequencies = np.ascontiguousarray([0.25, 0.25, 0.25, 0.25])
fixed_equal_nucleotide_frequencies.setflags(write=False)

                       #  0    1
binary_charmap = {'-': [1.0, 1.0],
                  'N': [1.0, 1.0],
                  '0': [1.0, 0.0],
                  '1': [0.0, 1.0]}

# Partials are in TCAG order, i.e. paml order
                    #  T    C    A    G
dna_charmap = {'-': [1.0, 1.0, 1.0, 1.0],
               'T': [1.0, 0.0, 0.0, 0.0],
               'C': [0.0, 1.0, 0.0, 0.0],
               'A': [0.0, 0.0, 1.0, 0.0],
               'G': [0.0, 0.0, 0.0, 1.0],
               'U': [1.0, 0.0, 0.0, 0.0],
               'R': [0.0, 0.0, 1.0, 1.0],
               'Y': [1.0, 1.0, 0.0, 0.0],
               'M': [0.0, 1.0, 1.0, 0.0],
               'K': [1.0, 0.0, 0.0, 1.0],
               'W': [1.0, 0.0, 1.0, 0.0],
               'S': [0.0, 1.0, 0.0, 1.0],
               'B': [1.0, 1.0, 0.0, 1.0],
               'D': [1.0, 0.0, 1.0, 1.0],
               'H': [1.0, 1.0, 1.0, 0.0],
               'V': [0.0, 1.0, 1.0, 1.0],
               'N': [1.0, 1.0, 1.0, 1.0],
               't': [1.0, 0.0, 0.0, 0.0],
               'c': [0.0, 1.0, 0.0, 0.0],
               'a': [0.0, 0.0, 1.0, 0.0],
               'g': [0.0, 0.0, 0.0, 1.0],
               'u': [1.0, 0.0, 0.0, 0.0],
               'r': [0.0, 0.0, 1.0, 1.0],
               'y': [1.0, 1.0, 0.0, 0.0],
               'm': [0.0, 1.0, 1.0, 0.0],
               'k': [1.0, 0.0, 0.0, 1.0],
               'w': [1.0, 0.0, 1.0, 0.0],
               's': [0.0, 1.0, 0.0, 1.0],
               'b': [1.0, 1.0, 0.0, 1.0],
               'd': [1.0, 0.0, 1.0, 1.0],
               'h': [1.0, 1.0, 1.0, 0.0],
               'v': [0.0, 1.0, 1.0, 1.0],
               'n': [1.0, 1.0, 1.0, 1.0]}

                        #  A    R    N    D    C    Q    E    G    H    I    L    K    M    F    P    S    T    W    Y    V
protein_charmap = {'-': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                   '?': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                   'A': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   'R': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   'N': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   'D': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   'C': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   'Q': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   'E': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   'G': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   'H': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   'I': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   'L': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   'K': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   'M': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   'F': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   'P': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   'S': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                   'T': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                   'W': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                   'Y': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                   'V': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                   'X': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                   'a': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   'r': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   'n': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   'd': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   'c': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   'q': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   'e': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   'g': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   'h': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   'i': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   'l': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   'k': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   'm': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   'f': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   'p': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   's': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                   't': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                   'w': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                   'y': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                   'v': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                   'x': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}
