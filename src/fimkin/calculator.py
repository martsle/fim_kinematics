from typing import List, Tuple
from numpy.typing import NDArray
from numpy import float64

Vector3D = List[float]
Vector6D = List[float]

class KinematicsCalculator:
    def __init__(self, robot:int = 2, threshold:float = 0.1) -> None:
        pass
        
    def setDesiredPOE(self, pose:Vector6D) -> None:
        pass

    def forward(self, theta:Vector6D) -> NDArray[float64]:
        pass

    def inverse_all(self):
        pass

    def inverse(self, prev_pose:Vector6D = [0, 0, 0, 0, 0, 0]) -> Tuple[float]:
        pass