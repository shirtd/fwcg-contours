import numpy as np
import os, sys

DATA_DIR = 'data'
DATA = {'rainier' : os.path.join(DATA_DIR, 'rainier16.dat'),
        'cascades' : os.path.join(DATA_DIR, 'cascades16.dat'),
        'seattle' : os.path.join(DATA_DIR, 'seattle16.dat')}

COLOR_ORDER = ['blue','green','yellow','salmon','purple']
CUTS = [0.0, 0.15, 0.28, 0.38, 0.48, 1.0]
LABELS = ['A','B','C','D','E']

ON_ALPHA = 1
CULLING = False
LIGHTING = True
STRETCH = 4
PAD = 0

CUT_ARGS = {'A' : {'min' : CUTS[0], 'max' : CUTS[1],    'color' : COLOR[COLOR_ORDER[0]]},
            'B' : {'min' : CUTS[1], 'max' : CUTS[2],    'color' : COLOR[COLOR_ORDER[1]]},
            'C' : {'min' : CUTS[2], 'max' : CUTS[3],    'color' : COLOR[COLOR_ORDER[2]]},
            'D' : {'min' : CUTS[3], 'max' : CUTS[4],    'color' : COLOR[COLOR_ORDER[3]]},
            'E' : {'min' : CUTS[4], 'max' : CUTS[5],    'color' : COLOR[COLOR_ORDER[4]]}}

SURF_ARGS = {   'A' : {'opacity' : ON_ALPHA, 'backface_culling' : CULLING,   'lighting' : LIGHTING},
                'B' : {'opacity' : ON_ALPHA, 'backface_culling' : CULLING,   'lighting' : LIGHTING},
                'C' : {'opacity' : ON_ALPHA, 'backface_culling' : CULLING,   'lighting' : LIGHTING},
                'D' : {'opacity' : ON_ALPHA, 'backface_culling' : CULLING,   'lighting' : LIGHTING},
                'E' : {'opacity' : ON_ALPHA, 'backface_culling' : CULLING,   'lighting' : LIGHTING}}

CONT_ARGS = {"%s_c" % s : {'scalar' : [CUTS[i]], 'color' : COLOR[COLOR_ORDER[i]]} for i,s in enumerate(LABELS)}

PADS = {'rainier' : 0, 'seattle' : 0, 'cascades' : 10}

STRETCHES = {'rainier' : 4, 'seattle' : 8, 'cascades' : 8}

VIEWS = {   'rainier' : {   'default'   : 'side',
                            'side'      : { 'view' : (29.71, 70.6, 10.7, np.array([ 0., -0.011,  0.386])),
                                            'zoom' : 4.6, 'roll' : -101.887, 'parallel_projection' : False},
                            'top'       : { 'view' : (0.0, 0.0, 6., np.array([0. , 0. , 0.5])),
                                            'zoom' : 2.6, 'roll' : -90, 'parallel_projection' : True}},
            'seattle' : {  'default'   : 'side',
                            'side'      : { 'view' : (39.73, 71.69, 25.91457, np.array([-0.3924,  0.4974,  0.4513])),
                                            'zoom' : 4.6, 'roll' : -100, 'parallel_projection' : False},
                            'top'       : { 'view' : (0.0, 0.0, 6., np.array([0. , 0. , 0.5])),
                                            'zoom' : 6.43, 'roll' : -90, 'parallel_projection' : True}},
            'cascades' : {  'default'   : 'side',
                            'side'      : { 'view' : (39.73, 71.69, 17.7, np.array([ 0.   , 0,  0.5])),
                                            'zoom' : 4.6, 'roll' : -100, 'parallel_projection' : False},
                            'top'       : { 'view' : (0.0, 0.0, 6., np.array([0. , 0. , 0.5])),
                                            'zoom' : 4.4, 'roll' : -90, 'parallel_projection' : True}}}
