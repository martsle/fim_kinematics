from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from .calculator import KinematicsCalculator, Vector6D

class Screwkincalc(KinematicsCalculator):
    """
    A kinematics calculator for robot arm control using Product of Exponentials (POE) approach.
    
    This class implements forward and inverse kinematics calculations for UR series robots
    (UR5e and UR10e) based on screw theory. It represents the robot's kinematic chain
    using twist coordinates and exponential maps for rigid body transformations.
    
    Attributes:
        robot (int): The robot model (1 for UR5e or 2 for UR10e).
        threshold (float): Angular threshold for joint angle continuity checks.
        DoF (int): Degrees of freedom of the robot.
        protective_stop (bool): Flag indicating if joint limits are exceeded.
        desiredPOE (NDArray): Target end-effector pose as a 4x4 transformation matrix.
        exceededBy (Tuple[float]): Magnitude of threshold violations.
        poseNo (int): Counter for tracking pose calculations.
        theta_old (Vector6D): Previous joint angles for continuity checking.
        a (NDArray): Location vectors for each joint in the kinematic chain.
        vel (NDArray): Unit twist axis for each joint.
        Sv (list): Moment vectors for each joint.
        M (NDArray): End-effector configuration in the home position.
        invM (NDArray): Inverse of the home position matrix.
        limits (list): Joint angle limits for each joint.
        
    Robot Dimensions:
        W1, W2: Wrist link dimensions
        L1, L2: Arm link lengths
        H1, H2: Height offsets
    """
    def __init__(self, robot: int = 2, threshold: float = 0.1, toolOffset: Optional[list[float]] = None) -> None:
        self.robot = robot
        self.threshold = threshold
        self.DoF:int = 5
        self.protective_stop:bool = False
        self.desiredPOE = np.eye(4)
        self.exceededBy: Tuple[float] = (0, 0, 0)
        self.poseNo:int = 0
        if toolOffset is not None and len(toolOffset) != 3:
            raise ValueError(f"toolOffset must have exactly 3 elements, got {len(toolOffset)}")
        self._toolOffset: list[float] = toolOffset if toolOffset is not None else [0.0, 0.0, 0.3798] #0.3575 + 0.0223 # measured with robot. TODO: all offset values should be used!!

        if self.robot == 1:   #  RobotType.UR5e
            self.W1 = 0.1333  #  J4
            self.W2 = 0.0996 + self._toolOffset[2]  #  J6
            self.L1 = 0.425   # -J2
            self.L2 = 0.3922  # -J3
            self.H1 = 0.1625  #  J1
            self.H2 = 0.0997  #  J5
        elif self.robot == 2: #  RobotType.UR10e
            self.W1 = 0.17415 #  J4
            self.W2 = 0.11655 + self._toolOffset[2] #  J6
            self.L1 = 0.6127  # -J2
            self.L2 = 0.57155 # -J3
            self.H1 = 0.1807  #  J1
            self.H2 = 0.11985 #  J5
        else:
            print('fail')
            return
            #raise KeyError

        self.theta_old: Vector6D = [0, 0, 0, 0, 0, 0]
        self.a = np.array([[0, 0, 0], 
                           [0, 0, self.H1],
                           [-self.L1, 0, self.H1], 
                           [-self.L1-self.L2, 0, self.H1], 
                           [-self.L1-self.L2, -self.W1, 0], 
                           [-self.L1-self.L2, 0, self.H1-self.H2]])
        self.vel = np.array([[0, 0, 1], 
                             [0, -1, 0], 
                             [0, -1, 0], 
                             [0, -1, 0], 
                             [0, 0, -1], 
                             [0, -1, 0]])
        self.Sv = [-np.cross(v,a) for v,a in zip(self.vel,self.a)]
        self.M = np.array([[1, 0, 0, -self.L1-self.L2], 
                           [0, 0, -1, -self.W1-self.W2], 
                           [0, 1, 0, self.H1-self.H2], 
                           [0, 0, 0, 1]])
        self.invM = self.invertPOE(self.M)
        self.limits = [[-2*np.pi, 2*np.pi],
                       [-2*np.pi, 2*np.pi],
                       [-2*np.pi, 2*np.pi],
                       [-2*np.pi, 2*np.pi],
                       [-2*np.pi, 2*np.pi],
                       [-2*np.pi, 2*np.pi]]

    def setDesiredPOE(self, pose: NDArray[np.float64]) -> None:
        if pose.shape == (6,):
            self.desiredPOE = self.forward(pose)
        else:
            self.desiredPOE = pose

    def forward(self, theta: Vector6D) -> NDArray[np.float64]:
        T=self.POEall(theta)
        Tf=np.dot(T,self.M)
        # Tf[0:3, 0:3] = self.normalizeRot(Tf[0:3, 0:3])
        return Tf
    
    @staticmethod
    def normalizeRot(R: NDArray[np.float64]) -> NDArray[np.float64]:
        u, _, vt = np.linalg.svd(R)
        rotn = u @ vt
        if np.linalg.det(rotn) < 0:
            u[:, -1] *= -1
            rotn = u @ vt
        return rotn

    def POE(self, angle: float, idx: int, inverse: bool = False) -> NDArray[np.float64]:
        T=np.eye(4,4)
        vel_hat=np.array([[ 0,              -self.vel[idx,2],  self.vel[idx,1]], 
                          [ self.vel[idx,2],  0,              -self.vel[idx,0]], 
                          [-self.vel[idx,1],  self.vel[idx,0],  0             ]])
        vel_hat_squared = vel_hat@vel_hat
        e_vel_hat=np.eye(3,3)+vel_hat*np.sin(angle)+vel_hat_squared*(1-np.cos(angle))
        t=(np.eye(3,3)*angle+(1-np.cos(angle))*vel_hat+(angle-np.sin(angle))*vel_hat_squared)@self.Sv[idx]

        if inverse:
            T[0:3,0:3] =  np.transpose(e_vel_hat)
            T[0:3,3]   = -np.transpose(e_vel_hat)@t
        else:
            T[0:3,0:3] = e_vel_hat
            T[0:3,3]   = t
        
        return T

    def POEall(self, theta: Vector6D) -> NDArray[np.float64]:
        T=np.eye(4,4)
        for ii in np.arange(self.DoF,-1,-1):
            vel_hat=np.array([[ 0,              -self.vel[ii,2],  self.vel[ii,1]], 
                              [ self.vel[ii,2],  0,              -self.vel[ii,0]], 
                              [-self.vel[ii,1],  self.vel[ii,0],  0             ]])

            vel_hat_squared = vel_hat@vel_hat
            e_vel_hat=np.eye(3,3)+vel_hat*np.sin(theta[ii])+vel_hat_squared*(1-np.cos(theta[ii]))
            
        
            if ii>0:
                Sv=-np.cross(self.vel[ii], self.a[ii])
            elif ii==0:
                Sv=np.array([0, 0, 0])
            
            t=(np.eye(3,3)*theta[ii]+(1-np.cos(theta[ii]))*vel_hat+(theta[ii]-np.sin(theta[ii]))*vel_hat_squared)@Sv

            fr = np.concatenate((e_vel_hat,np.array([t]).T), axis=1)

            e_zai=np.concatenate((fr,np.array([[0,0,0,1]])))
        
            T=e_zai@T
        
        return T

    def inverse_all(self):
        pass

    def inverse(self, prev_pose: Vector6D = [0, 0, 0, 0, 0, 0]) -> Tuple[float]: # TODO: why not list?
        self.poseNo += 1 # TODO: what for?
        Q = self.desiredPOE[0:2,3] - self.W2*self.desiredPOE[0:2,2]
        MQ2 = Q[0]**2 + Q[1]**2
        p2_first = (self.W1**2 / MQ2) * Q
        p2_last  = (self.W1 * np.sqrt(MQ2 - self.W1**2) / MQ2) * np.array([-Q[1], Q[0]])
        p2 = p2_first + p2_last
        cos1 = np.dot([0,-1], p2) / np.linalg.norm(p2)
        # z-component of the 2-D cross product [0, -1] x p2. NumPy no
        # longer accepts two-dimensional vectors in np.cross.
        sin1 = p2[0] / np.linalg.norm(p2)
        theta1 = np.arctan2(sin1,cos1)

        test, theta1 = self.check_theta(prev_pose[0], theta1)

        if test:
            p2 = p2_first - p2_last
            cos1 = np.dot([0,-1],p2)/(np.linalg.norm(p2))
            sin1 = p2[0]/(np.linalg.norm(p2))

            theta1 = np.arctan2(sin1,cos1)
            test, theta1 = self.check_theta(prev_pose[0], theta1)

        if test:
            print(f"Warning: theta1 threshold exceeded by {self.exceededBy}!")
        if theta1 < self.limits[0][0] or theta1 > self.limits[0][1]:
            self.protective_stop = True
            print("Warning: Limits at theta1 exceeded!")

        T1 = self.desiredPOE@self.invM
        cos5 = -T1[0,1]*sin1 + T1[1,1]*cos1
        theta5 = np.arccos(cos5)
        test, theta5 = self.check_theta(prev_pose[4], theta5)

        if test:
            theta5 *= -1
            test, theta5 = self.check_theta(prev_pose[4], theta5)
        
        if test:
            print(f"Warning: theta5 threshold exceeded by {self.exceededBy}!")
        if theta5 < self.limits[4][0] or theta5 > self.limits[4][1]:
            self.protective_stop = True
            print("Warning: Limits at theta5 exceeded!")
        
        sin5 = np.sin(theta5)
        if sin5 >=0:
            f =  1
        else:
            f = -1

        theta6 = np.arctan2(-f*T1[0,2]*np.sin(theta1) + f*T1[1,2]*np.cos(theta1),f*T1[0,0]*np.sin(theta1) - f*T1[1,0]*np.cos(theta1))
        test, theta6 = self.check_theta(prev_pose[5], theta6, 1)
        
        if test:
            print(f"Warning: theta6 threshold exceeded by {self.exceededBy}!")
        if theta6 < self.limits[5][0] or theta6 > self.limits[5][1]:
            self.protective_stop = True
            print("Warning: Limits at theta6 exceeded!")

        invT1 = self.POE(theta1, 0, True)
        invT5 = self.POE(theta5, 4, True)
        invT6 = self.POE(theta6, 5, True)
        T3 = invT1@T1@invT6@invT5

        sin234 = T3[2,0]
        cos234 = T3[0,0]

        theta234 = np.arctan2(sin234,cos234)
        test, theta234 = self.check_theta(prev_pose[1]+prev_pose[2]+prev_pose[3], theta234, 2)
        if test:
            print(f"Warning: theta234 threshold exceeded by {self.exceededBy}!")
        
        m = -T3[0,3] + (self.L1 + self.L2)*cos234 + self.H1*sin234
        n = -T3[2,3] + self.H1 - self.H1*cos234 + (self.L1 + self.L2)*sin234
        
        cos3 = (m**2 + n**2 - self.L1**2 - self.L2**2) / (2*self.L1*self.L2)
        theta3 =  np.arccos(cos3)
        test, theta3 = self.check_theta(prev_pose[2], theta3)
        if test: 
            test, theta3 = self.check_theta(prev_pose[2], -theta3)
        if test:
            print(f"Warning: theta3 threshold exceeded by {self.exceededBy}!")
        if theta3 < self.limits[2][0] or theta3 > self.limits[2][1]:
            self.protective_stop = True
            print("Warning: Limits at theta3 exceeded!")

        sin3 = np.sin(theta3)

        b = self.L1+self.L2*cos3+(self.L2**2*sin3**2)/(self.L1+self.L2*cos3)
        sin2 = n/b - (self.L2*sin3*m)/(self.L1+self.L2*cos3)/b
        cos2 = m/b + (self.L2*sin3*n)/(self.L1+self.L2*cos3)/b

        theta2 = np.arctan2(sin2,cos2)
        test, theta2 = self.check_theta(prev_pose[1], theta2)
        if test:
            print(f"Warning: theta2 threshold exceeded by {self.exceededBy}!")
        if theta2 < self.limits[1][0] or theta2 > self.limits[1][1]:
            self.protective_stop = True
            print("Warning: Limits at theta2 exceeded!")
        
        theta4 = theta234 - theta3 - theta2
        if theta4 < self.limits[3][0] or theta4 > self.limits[3][1]:
            self.protective_stop = True
            print("Warning: Limits at theta234 exceeded!")
        
        return theta1, theta2, theta3, theta4, theta5, theta6
    
    @staticmethod
    def invertPOE(T: NDArray[np.float64]) -> NDArray[np.float64]:
        invT = np.eye(4)
        TT = np.transpose(T[0:3,0:3])
        invT[0:3,0:3] = TT
        invT[0:3,3] = -TT@T[0:3,3]
        return invT

    def check_theta(self, prev: float, new: float, factor: int = 2):
        newppi = new + factor*np.pi
        newmpi = new - factor*np.pi

        test1 = prev - new
        test2 = prev - newppi
        test3 = prev - newmpi

        if abs(test1) < self.threshold:
            return 0, new
        elif abs(test2) < self.threshold:
            return 0, newppi
        elif abs(test3) < self.threshold:
            return 0, newmpi
        else:
            self.exceededBy = (test1, test2, test3)
            return 1, new
        
        # variants = [new, new + factor*np.pi, new - factor*np.pi]
        # tests = [prev - new, prev - variants[1], prev - variants[2]]

        # idx = np.argmin(np.abs(tests))
        # if tests[idx] < self.threshold:
        #     return 0, variants[idx]
        # else:
        #     self.exceededBy = tests
        #     return 1, variants[idx]
