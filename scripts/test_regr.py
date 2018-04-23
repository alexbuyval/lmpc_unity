#!/usr/bin/env python

import rospy
import numpy as np
import math
from SysID import LocLinReg, Regression, EstimateABC
from FTOCP import BuildMatEqConst, BuildMatCost, BuildMatIneqConst, FTOCP, GetPred
from LMPC import LMPC, ComputeCost, LMPC_BuildMatEqConst, LMPC_BuildMatIneqConst
from cvxopt.solvers import qp
from cvxopt import spmatrix, matrix, solvers
from scipy import linalg

"""
states = np.array([[ 1.48883515,  0.,          0.,          0.,          0.50717646,  0.        ],
 [ 1.53324639,  0.,          0.,          0.,          0.53762376,  0.        ],
 [ 1.57765562,  0.,          0.,          0.,          0.56895971,  0.        ],
 [ 1.62206284,  0.,          0.,          0.,         0.60118389 , 0.        ],
 [ 1.66646791,  0.,          0.,          0.,          0.63429624,  0.        ],
 [ 1.73307136,  0.,          0.,          0.,         0.68562901,  0.        ]] )
"""
states = np.array([[ 1.36,  0.,    0.,    0.,    0.5,   0.  ],
 [ 1.37,  0.,    0.,    0.,    0.53,  0.  ],
 [ 1.38,  0.,    0.,    0.,    0.55,  0.  ],
 [ 1.39,  0.,    0.,    0.,    0.58,  0.  ],
 [ 1.4,  0.,    0.,    0.,    0.61,  0.  ],
 [ 1.41,  0.,    0.,    0.,    0.64,  0.  ],
 [ 1.43,  0.,    0.,    0.,    0.67,  0.  ]])

u = np.array([[ 0.,   0.5],
 [ 0.,   0.5],
 [ 0.,   0.5],
 [ 0.,   0.5],
 [ 0.,   0.5],
 [ 0.,   0.5]]
)
    
lamb = 0.0000001
print "X: \n", states, "\n U: ", u
A, B = Regression(states, u, lamb)
print "A matrix: \n", A, "\n B matrix: \n", B
