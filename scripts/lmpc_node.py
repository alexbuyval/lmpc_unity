#!/usr/bin/env python

import rospy
import numpy as np
import math
from FTOCP import BuildMatEqConst, BuildMatCost, BuildMatIneqConst, FTOCP, GetPred
from SysID import LocLinReg, Regression, EstimateABC, LMPC_EstimateABC
from LMPC import LMPC, ComputeCost, LMPC_BuildMatEqConst, LMPC_BuildMatIneqConst
from Track import CreateTrack, Evaluate_e_ey
from cvxopt.solvers import qp
from cvxopt import spmatrix, matrix, solvers
from scipy import linalg
from numpy import linalg as la
import datetime

from car_mpc_control.msg import LogMPC
from car_unity_simulator.msg import CarState, CarControl
import csv

solvers.options['show_progress'] = False      # Turn off CVX messages

Points = 900
N = 8

u = np.zeros((Points, 2))          # Initialize the input vector
states = np.zeros((Points, 6))   # Initialize state vector 
LinPoints = np.zeros((N+1, 6))
 
i = 0
inter_step = 2.0
x, y, cur, tang, x_in, y_in = [], [], [], [], [], []

last_throttle = 0.0
last_steer = 0.0

ind = 0
ey = 0.0
ePsi = 0.0
vx = 0.0
vy = 0.0
dPsi = 0.0
distAlong = 0.0

init_fist_state = False

identification_done = False
firstMPCstep = True

Lap = 0
last_dist = 0

Laps = 10

numSS_Points = 30

TimeSS= 10000*np.ones(Laps+2)
addSS = np.zeros(Laps+2)
SS    = 10000*np.ones((2*states.shape[0], 6, Laps+2))
uSS   = 10000*np.ones((2*states.shape[0], 2, Laps+2))
Qfun  = 0*np.ones((2*states.shape[0], Laps+2)) # Need to initialize at zero as adding point on the fly

additionalPoints = 0

#time statistics
mean_lin_time = 0.0
max_lin_time = 0.0
min_lin_time = 9999.0
mean_solve_time = 0.0
max_solve_time = 0.0
min_solve_time = 9999.0
sum_lin_time = 0.0
sum_solve_time = 0.0
count_lin_time = 0
count_solve_time = 0

"""
A = np.zeros((6,6))
B = np.zeros((6,2))

"""
A = np.array([[  9.98e-01,  -2.40e-03,  -4.48e-03,   1.31e-03,   1.53e-06,  -3.31e-04],
 [  5.22e-05,   9.08e-01,  -5.80e-02,  -6.71e-05,   4.75e-07,  -2.42e-04],
 [  3.36e-04,  -4.47e-02,   7.44e-01,  -6.30e-04,  -1.10e-06,  -7.48e-04],
 [  2.13e-04,   1.51e-03,   1.69e-02,   1.00e+00,   3.28e-07,   1.41e-03],
 [  1.98e-02,  -7.05e-03,   2.12e-03,  -3.51e-03,   1.00e+00,  -3.37e-03],
 [  2.67e-05,   1.79e-02,   5.44e-04,   7.87e-02,   1.78e-07,   1.00e+00],])

B = np.array([[  1.02e-02,   1.69e-02],
 [  1.44e-01,  -6.89e-04],
 [  3.54e-01,  -2.17e-03],
 [  3.66e-03,  -3.38e-04],
 [ -1.55e-04,   1.89e-04],
 [  1.49e-03,  -1.24e-04]]) 

def load_track():
    global x, y, cur, tang
    with open('/home/alex/catkin_ws/src/lmpc_unity/scripts/lmpc_track.csv') as f:
        track_data = csv.reader(f, delimiter=',')
        for row in track_data:
            x.append(float(row[0]))
            y.append(float(row[1]))
            cur.append(float(row[3]))
            tang.append(float(row[2]))
    return
            

def nearest_point(car_pos, x, y):

    distances = [np.sqrt((car_pos[0] - xi)**2 + (car_pos[1] - yi)**2) for xi, yi in zip(x, y)]
    index = distances.index(min(distances))

    return (index, x[index], y[index])

def min_dist(pt, x, y):
    """Return minimum distance from pt to a point in arrys x,y"""
    return (min([np.sqrt((pt[0]-xi)**2 + (pt[1]-yi)**2) for xi, yi in zip(x,y)]))


def callback_carstate(msg):

  global x, y, cur, tang
  global ind, ey, ePsi, vx, vy, distAlong, dPsi


  trackLength = (len(x)-1)*inter_step

  car_pos = [msg.pos.x, msg.pos.y]
  ind, x_n, y_n = nearest_point(car_pos, x, y)
  # % calculate angular error
  ePsi = msg.orientation.z - tang[ind]
  ePsi = math.atan2(math.sin(ePsi), math.cos(ePsi))

  # % calculate lateral error
  q1=np.array([x[ind], y[ind]])
  nxt = (ind + 1) % len(x)
  q2 = np.array([x[nxt], y[nxt]])

  r = ((car_pos-q2).dot(q1 - q2)*q1 + (car_pos-q1).dot(q2-q1)*q2)/(q2-q1).dot(q2-q1)
  ey = np.linalg.norm(r - car_pos)

  dq = q2 - q1
  dpos = car_pos - q1
  c = np.cross(np.array([dq[0], dq[1], 0]), np.array([dpos[0], dpos[1], 0]))
  ey = ey*np.sign(c[2])
  
  locDistAlong = np.linalg.norm(r - q1)
  distAlongNext = np.linalg.norm(r - q2) 

  if distAlongNext > inter_step:
    locDistAlong = - locDistAlong

  if ind==0 and locDistAlong<0: #it happens when car crossed the start line
    distAlong = trackLength + locDistAlong
  else:
    distAlong = locDistAlong + ind*inter_step


  vel = math.sqrt(msg.vel.x*msg.vel.x + msg.vel.y*msg.vel.y)
  slipAngle = math.atan2(msg.vel.y, msg.vel.x) - msg.orientation.z          
  slipAngle = math.atan2(math.sin(slipAngle), math.cos(slipAngle))

  vx = vel*math.cos(slipAngle)
  vy = vel*math.sin(slipAngle)

  dPsi = msg.angular_vel.z
  #rospy.loginfo("State: vx %s vy %s dPsi %s ePsi %s s %s ey %s", vx, vy, dPsi, ePsi, distAlong, ey)
  return


def collectData():
  global x
  global u
  global Points
  global i
  global ind, ey, ePsi, vx, vy, distAlong, dPsi
  global identification_done
  global N
  global LinPoints
  global min_lin_time, mean_lin_time, max_lin_time, min_solve_time, mean_solve_time, max_solve_time

  if i<Points:
    states[i,0] = vx
    states[i,1] = vy
    states[i,2] = dPsi
    states[i,3] = ePsi
    states[i,4] = distAlong
    states[i,5] = ey

    u[i,0] = last_steer
    u[i,1] = last_throttle

    #print "i: ",i, " ind:", ind, " State: ", states[i,:], " Control: ", u[i,:]
    i=i+1
  #else:
    
    #lamb = 0.01
    #print "X: \n", states, "\n U: ", u
    #A, B = Regression(states, u, lamb)
    #print "A matrix: \n", A, "\n B matrix: \n", B
    
    #LinPoints = states[ind:ind+N+1,:]
    #identification_done = True

def newLap():
  global TimeSS, uSS, SS, Qfun, Lap, i
  global states, u
  global firstMPCstep
  global additionalPoints
  global sum_lin_time, sum_solve_time, count_lin_time, count_solve_time
  global min_lin_time, max_lin_time, min_solve_time, max_solve_time

  TimeSS[Lap] = i
  SS[0:TimeSS[Lap],:, Lap]  = states[0:i,:]
  uSS[0:TimeSS[Lap],:, Lap]  = u[0:i,:]
  Qfun[0:TimeSS[Lap], Lap] = ComputeCost(states[0:i,:], u[0:i,:], np, TrackLength) 
  print "Cost at lap ", Lap, " is ",Qfun[0,Lap]

  Lap = Lap + 1
  firstMPCstep = True
  additionalPoints = 0
  i=0

  if count_lin_time>0:
    print "Linearization time. Min: ", min_lin_time, " Mean: ", sum_lin_time/count_lin_time, " Max: ", max_lin_time
  if count_solve_time>0:
    print "Solving time. Min: ", min_solve_time, " Mean: ", sum_solve_time/count_solve_time, " Max: ", max_solve_time

  #reset time statistics
  max_lin_time = 0.0
  min_lin_time = 9999.0
  max_solve_time = 0.0
  min_solve_time = 9999.0
  sum_lin_time = 0.0
  sum_solve_time = 0.0
  count_lin_time = 0
  count_solve_time = 0

# main function
if __name__ == '__main__':
	# initialize node
  rospy.init_node('lmpc_node')
  load_track() #we need to combine both tracks
  PointAndTangent, TrackLength = CreateTrack()
	# Subscribers
  car_state_sub = rospy.Subscriber("/car_state", CarState, callback_carstate)

  cmd_pub = rospy.Publisher("/control", CarControl, queue_size = 10)

  Hz = 10.0
  dt = 1.0/Hz
  loop_rate = rospy.Rate(Hz)

  target_speed = 5.0
  n = 6
  d = 2

  swifth = N-1
  
  Q = np.diag([5.0, 0.0, 0, 1, 0.0, 10.0]) # vx, vy, wz, epsi, s, ey
  R = np.diag([30.0, 5.0]) # delta, a
  F, b = BuildMatIneqConst(N, n, np, linalg, spmatrix)
  M, q    = BuildMatCost(Q, R, Q, N, linalg, np, spmatrix, target_speed)
  
  F_LMPC, b_LMPC = LMPC_BuildMatIneqConst(N, n, np, linalg, spmatrix, numSS_Points)

  Qslack = 50*np.diag([1, 10, 10, 10, 10, 10])
  Q_LMPC = 0 * np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # vx, vy, wz, epsi, s, ey
  R_LMPC = 5 * np.diag([10.0, 1.0])  # delta, a



  while not rospy.is_shutdown():

    if distAlong<last_dist-10.0: #it means that car cross start line
      newLap()
    
    if vx>0.5:
      collectData()
    
    x0 = np.array([vx, vy, dPsi, ePsi, distAlong, ey ])
    #print "\n x0: ", x0

    last_dist = distAlong
    if Lap==0:

      last_steer = - 0.1 * ey - 0.1 * ePsi + np.maximum(-0.1, np.min(np.random.randn()*0.05, 0.1))
      last_throttle = 0.5*(target_speed - vx) + np.maximum(-0.1, np.min(np.random.randn()*0.05, 0.1))

    elif Lap==1:

      startTimer = datetime.datetime.now() # Start timer for LMPC iteration
      if firstMPCstep:
        G, E, L = BuildMatEqConst(A, B, np.zeros((n, 1)), N, n, d, np, spmatrix, 0)
        firstMPCstep = False
      else:
        Atv, Btv, Ctv = EstimateABC(LinPoints, N, n, d, states, u, qp, matrix, PointAndTangent, dt)
        G, E, L = BuildMatEqConst(Atv, Btv, Ctv, N, n, d, np, spmatrix, 1)
      endTimer = datetime.datetime.now(); deltaTimer_tv = endTimer - startTimer

      startTimer = datetime.datetime.now() # Start timer for LMPC iteration
      Sol, feasible = FTOCP(M, q, G, L, E, F, b, x0, np, qp, matrix)
      endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer

      if i<3:
        print("Linearization time: %.4fs Solver time: %.4fs" % (deltaTimer_tv.total_seconds(), deltaTimer.total_seconds()))

      xPred, uPred = GetPred(Sol, n, d, N, np)
      LinPoints = xPred.T
      LinInput = uPred.T
      #print "\n xPred: ", xPred, "\n uPred:", uPred

      last_steer = np.asscalar(uPred[0,0])
      last_throttle = np.asscalar(uPred[1,0])
      #print "\n Steer: ", last_steer, "Throttle: ", last_throttle
    else:

      startTimer = datetime.datetime.now()  # Start timer for LMPC iteration

      Atv, Btv, Ctv, indexUsed_list = LMPC_EstimateABC(LinPoints, LinInput, N, n, d, SS, uSS, TimeSS, qp, matrix,
                                                                 PointAndTangent, dt, Lap)
      endTimer = datetime.datetime.now(); deltaTimer_tv = endTimer - startTimer
      G, E, L, npG, npE = LMPC_BuildMatEqConst(Atv, Btv, Ctv, N, n, d, np, spmatrix, 1)
      
      Sol, feasible, deltaTimer, slack = LMPC(npG, L, npE, F_LMPC, b_LMPC, x0, np, qp, matrix, datetime, la, SS,
                                                    Qfun,  N, n, d, spmatrix, numSS_Points, Qslack, Q_LMPC, R_LMPC, Lap, swifth)
      
      #if i<10 or (i>30 and i<40):
      #  print("i=%s Linearization time: %.4fs Solver time: %.4fs" % (i, deltaTimer_tv.total_seconds(), deltaTimer.total_seconds()))
      #print "Sol: ", Sol
      #print "slack: ", slack

      lin_time = deltaTimer_tv.total_seconds()
      sum_lin_time = sum_lin_time + lin_time
      count_lin_time = count_lin_time + 1
      if lin_time < min_lin_time:
        min_lin_time = lin_time
      if lin_time>max_lin_time:
        max_lin_time = lin_time


      solve_time = deltaTimer.total_seconds()
      sum_solve_time = sum_solve_time + solve_time
      count_solve_time = count_solve_time + 1
      if solve_time < min_solve_time:
        min_solve_time = solve_time
      if solve_time>max_solve_time:
        max_solve_time = solve_time

      xPred, uPred = GetPred(Sol, n, d, N, np)
      #print "\n LMPC xPred: ", xPred, "\n uPred:", uPred
      LinPoints = np.vstack((xPred.T[1:,:], xPred.T[-1,:]))
      LinInput = uPred.T
      last_steer = np.asscalar(uPred[0,0])
      last_throttle = np.asscalar(uPred[1,0])
      
      addSS[Lap-1] = addSS[Lap-1] + 1
      indAdd = TimeSS[Lap-1] + addSS[Lap-1]
      SS[indAdd, :, Lap - 1]  = x0 + np.array([0, 0, 0, 0, TrackLength, 0])
      uSS[indAdd, :, Lap - 1] = np.array([last_steer, last_throttle])

      
      #print "\n Steer: ", last_steer, "Throttle: ", last_throttle

    control_msg = CarControl()
    control_msg.header.stamp = rospy.Time.now()
    control_msg.throttle = last_throttle
    control_msg.steering = math.degrees(last_steer)

    cmd_pub.publish(control_msg)
    #rospy.loginfo("Publish control: steer %s throttle %s", last_steer, last_throttle)
    loop_rate.sleep()
