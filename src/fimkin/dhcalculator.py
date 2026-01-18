import numpy as np
from numpy import linalg
import cmath
import math
from math import cos
from math import sin
from math import atan2
from math import acos
from math import asin
from math import sqrt
from math import pi
from .calculator import KinematicsCalculator

import logging
logger = logging.getLogger(__name__)

class DHkincalc(KinematicsCalculator):
  def __init__(self, robot: int = 2):
    self.robot = robot
    # Parameters source: https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
    if self.robot == 2:
      self.d = np.matrix([0.1807, 0, 0, 0.17415, 0.11985, 0.11655]) 
      self.a = np.matrix([0, -0.6127, -0.57155, 0, 0, 0])  
      self.alph = np.matrix([pi/2, 0, 0, pi/2, -pi/2, 0]) 
    elif self.robot == 1:
      self.d = np.matrix([0.1625, 0, 0, 0.1333, 0.0997, 0.0996]) 
      self.a = np.matrix([0, -0.425, -0.3922, 0, 0, 0]) 
      self.alph = np.matrix([pi/2, 0, 0, pi/2, -pi/2, 0])  
    # elif self.robot == "ur10":
    #   self.d = np.matrix([0.1273, 0, 0, 0.163941, 0.1157, 0.0922]) 
    #   self.a = np.matrix([0, -0.612, -0.5723, 0, 0, 0]) 
    #   self.alph = np.matrix([pi/2, 0, 0, pi/2, -pi/2, 0])  
    # elif self.robot == "ur5":
    #   self.d = np.matrix([0.089159, 0, 0, 0.10915, 0.09465, 0.0823]) 
    #   self.a = np.matrix([0 ,-0.425 ,-0.39225 ,0 ,0 ,0]) 
    #   self.alph = np.matrix([math.pi/2, 0, 0, math.pi/2, -math.pi/2, 0 ])  
    else:
      logging.error(f"Robot {robot} is not supported. Use UR5/e or UR10/e.")

    global d1, a2, a3, a7, d4, d5, d6

    d1 = 0.1625
    a2 = -0.425
    a3 = -0.3922
    #a7 = 0.075
    d4 = 0.1333
    d5 = 0.0997
    d6 = 0.0996


  def AH(self, n, th, c):

    T_a = np.matrix(np.identity(4), copy=False)
    T_a[0, 3] = self.a[0, n-1]
    T_d = np.matrix(np.identity(4), copy=False)
    T_d[2, 3] = self.d[0, n-1]

    Rzt = np.matrix([[cos(th[n-1, c]), -sin(th[n-1, c]), 0, 0],
              [sin(th[n-1, c]),  cos(th[n-1, c]), 0, 0],
              [0,               0,              1, 0],
              [0,               0,              0, 1]], copy=False)

    Rxa = np.matrix([[1, 0,                 0,                  0],
              [0, cos(self.alph[0, n-1]), -sin(self.alph[0, n-1]),   0],
              [0, sin(self.alph[0, n-1]),  cos(self.alph[0, n-1]),   0],
              [0, 0,                 0,                  1]], copy=False)

    A_i = T_d * Rzt * T_a * Rxa

    return A_i

  def forward(self, th, c = [0]):
    if isinstance(th, list):
      th = np.matrix([[th[0]], [th[1]], [th[2]], [th[3]], [th[4]], [th[5]]])
    elif isinstance(th, np.ndarray):
      if len(th.shape) == 1:
        th = np.matrix([[th[0]], [th[1]], [th[2]], [th[3]], [th[4]], [th[5]]])
      else:
        th = np.matrix(th)
    A_1 = self.AH(1, th, c)
    A_2 = self.AH(2, th, c)
    A_3 = self.AH(3, th, c)
    A_4 = self.AH(4, th, c)
    A_5 = self.AH(5, th, c)
    A_6 = self.AH(6, th, c)

    T_06 = A_1*A_2*A_3*A_4*A_5*A_6

    return T_06

  def inverse(self, desired_pos):  # T60
    th = np.matrix(np.zeros((6, 8)))
    P_05 = (desired_pos * np.matrix([0, 0, -d6, 1]).T-np.matrix([0, 0, 0, 1]).T)

    # **** theta1 ****

    psi = atan2(P_05[2-1, 0], P_05[1-1, 0])
    phi = acos(min(1.0, d4 / sqrt(P_05[2-1, 0]*P_05[2-1, 0] + P_05[1-1, 0]*P_05[1-1, 0])))
    #The two solutions for theta1 correspond to the shoulder
    #being either left or right
    th[0, 0:4] = pi/2 + psi + phi
    th[0, 4:8] = pi/2 + psi - phi
    th = th.real

    # **** theta5 ****

    cl = [0, 4]  # wrist up or down
    for i in range(0, len(cl)):
        c = cl[i]
        T_10 = linalg.inv(self.AH(1, th, c))
        T_16 = T_10 * desired_pos
        th[4, c:c+2] = + acos((T_16[2, 3]-d4)/d6)
        th[4, c+2:c+4] = - acos((T_16[2, 3]-d4)/d6)

    th = th.real

    # **** theta6 ****
    # theta6 is not well-defined when sin(theta5) = 0 or when T16(1,3), T16(2,3) = 0.

    cl = [0, 2, 4, 6]
    for i in range(0, len(cl)):
        c = cl[i]
        T_10 = linalg.inv(self.AH(1, th, c))
        T_16 = linalg.inv(T_10 * desired_pos)
        th[5, c:c+2] = atan2((-T_16[1, 2]/sin(th[4, c])),
                            (T_16[0, 2]/sin(th[4, c])))

    th = th.real

    # **** theta3 ****
    cl = [0, 2, 4, 6]
    for i in range(0, len(cl)):
        c = cl[i]
        T_10 = linalg.inv(self.AH(1, th, c))
        T_65 = self.AH(6, th, c)
        T_54 = self.AH(5, th, c)
        T_14 = (T_10 * desired_pos) * linalg.inv(T_54 * T_65)
        P_13 = T_14 * np.matrix([0, -d4, 0, 1]).T - np.matrix([0, 0, 0, 1]).T
        t3 = cmath.acos((linalg.norm(P_13)**2 - a2**2 - a3**2) /
                        (2 * a2 * a3))  # norm ?
        th[2, c] = t3.real
        th[2, c+1] = -t3.real

    # **** theta2 and theta 4 ****

    cl = [0, 1, 2, 3, 4, 5, 6, 7]
    for i in range(0, len(cl)):
        c = cl[i]
        T_10 = linalg.inv(self.AH(1, th, c))
        T_65 = linalg.inv(self.AH(6, th, c))
        T_54 = linalg.inv(self.AH(5, th, c))
        T_14 = (T_10 * desired_pos) * T_65 * T_54
        P_13 = T_14 * np.matrix([0, -d4, 0, 1]).T - np.matrix([0, 0, 0, 1]).T

        # theta 2
        th[1, c] = -atan2(P_13[1], -P_13[0]) + \
            asin(a3 * sin(th[2, c])/linalg.norm(P_13))
        # theta 4
        T_32 = linalg.inv(self.AH(3, th, c))
        T_21 = linalg.inv(self.AH(2, th, c))
        T_34 = T_32 * T_21 * T_14
        th[3, c] = atan2(T_34[1, 0], T_34[0, 0])
    th = th.real

    return th