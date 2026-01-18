from typing import List, Optional
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .screwcalculator import Screwkincalc

class Waypoint:
    _kinematicSolver = None
    def __init__(self, coord: NDArray[np.float64] = np.zeros((3,)), normal: NDArray[np.float64] = np.array((0.0, 0.0, -1.0)), velocity: float = 0.0) -> None:
        self._jointangles: NDArray[np.float64]  = np.zeros((6,))
        self._targetNormal: NDArray[np.float64] = normal
        self._Tf: NDArray[np.float64]           = np.eye(4)
        self._Tf[0:3, 3]                        = coord
        self._uv: NDArray[np.float64]           = np.zeros((2,))
        self._velocity: float                   = velocity
    
    #region properties
    @property
    def cartesian(self) -> NDArray[np.float64]:
        return self._Tf[0:3, 3]
    
    @property
    def jointangles(self) -> NDArray[np.float64]:
        return self._jointangles
    
    @property
    def normal(self) -> NDArray[np.float64]:
        return self._Tf[0:3, 2]

    @property
    def rot(self) -> NDArray[np.float64]:
        return self._Tf[0:3, 0:3]
    
    @property
    def RPY(self) -> NDArray[np.float64]:
        r = self.rot
        beta = np.arctan2(-r[3,1],np.sqrt(r[1,1]**2 + r[2,1]**2))
        alpha = np.arctan2(r[2,1]/np.cos(beta), r[1,1]/np.cos(beta))
        gamma = np.arctan2(r[3,2]/np.cos(beta), r[3,3]/np.cos(beta))
        return np.array([gamma, beta, alpha])
    
    @property
    def Tf(self) -> NDArray[np.float64]:
        return self._Tf
    
    @property
    def pose(self) -> NDArray[np.float64]:
        return np.concatenate((self.cartesian, self.RPY))
    
    @property
    def poseD(self) -> NDArray[np.float64]:
        rpy = self.RPY * 180.0 / np.pi
        return np.concatenate((self.cartesian, rpy))
    #endregion
    
    #region methods  
    def inverse(self, previous: NDArray[np.float64], solver: 'Screwkincalc') -> NDArray[np.float64]:
        if not np.any(self._jointangles):
            solver.setDesiredPOE(self.Tf)
            nextJC = solver.inverse(previous)
            self._jointangles[:] = nextJC
        else:
            print("warning: joint angles already set")
            #logger.warning("Kinematic Solver: There is already joint angles available. Skipping inverse calculation.")
            nextJC = self._jointangles
        return nextJC
    
    def _forward(self, solver: 'Screwkincalc') -> None:
        self._Tf = solver.forward(self._jointangles)
    #endregion

    #region public
    def getJointangles(self) -> NDArray[np.float64]:
        return self._jointangles      
    
    def rotByNormalAndReference(self, refRot: NDArray[np.float64], targetNormal:Optional[NDArray[np.float64]]=None) -> None:
        if targetNormal is not None:
            self._targetNormal = targetNormal
        c = np.dot(refRot[0:3,2], self._targetNormal) # assuming both are normalized
        if c == 1:
            self._Tf[0:3, 0:3] = refRot
        elif c == -1:
            raise NotImplementedError("Rotation by 180 degrees is not implemented yet.") # should not happen in practice
        else:
            k = np.cross(refRot[0:3,2], self._targetNormal)
            s = np.linalg.norm(k)
            k = k / s # normalizing rotational axis!
            vSkew = np.array([[0, -k[2], k[1]],[k[2], 0, -k[0]],[-k[1], k[0], 0]])
            R = np.eye(3) + s * vSkew + (1 - c) * vSkew@vSkew # Rodrigues formula
            self._Tf[0:3, 0:3] = self.normalizeRot(R@refRot)
        return self._Tf[0:3, 0:3]
    
    def setNormal(self, normal: NDArray[np.float64]) -> None:
        self._targetNormal = normal

    def setRot(self, rot: NDArray[np.float64]) -> None:
        self._Tf[0:3, 0:3] = rot
    
    def setXYZ(self, xyz: NDArray[np.float64]) -> None:
        self._Tf[0:3, 3] = xyz
    #endregion

    #region static methods
    @staticmethod
    def normalizeRot(rot) -> None:
        u, _, vh = np.linalg.svd(rot)
        rotn = u @ vh
        if np.linalg.det(rotn) < 0:
            u[:, -1] *= -1
            rotn = u @ vh
        return rotn
    
    @staticmethod
    def projectVec2RotPlane(vec: NDArray[np.float64], rot: NDArray[np.float64], rdir: int) -> NDArray[np.float64]:
        return vec - (np.dot(vec, rot[0:3, rdir]) /np.linalg.norm(rot[0:3, rdir]) * rot[0:3, rdir])
    #endregion

    #region class methods
    @classmethod
    def byJC(cls, joint: List[float], solver: 'Screwkincalc') -> 'Waypoint':
        obj = cls()
        obj._jointangles = np.array(joint)
        obj._forward(solver)
        return obj
    
    @classmethod
    def newListByCoordArray(cls, coords: NDArray[np.float64], normal: Optional[NDArray[np.float64]], velocity: Optional[NDArray[np.float64]]): # TODO: loose OCC dependency!
        assert coords.shape[0] == 3, "Input array must have shape (3,N)"
        assert coords.ndim == 2, "Input array must be 2-dimensional"
        if normal is None:
            normal = np.zeros((3, coords.shape[1]))
            normal[2,:] = -1.0
        if velocity is None:
            velocity = np.zeros((coords.shape[1],))
        return [cls(coord=coords[:,i], normal=normal[:,i], velocity=velocity[i]) for i in range(coords.shape[1])]
    #endregion
    #endregion

class Path():
    def __init__(self) -> None:
        self._segments: 'List[PathSegment]' = []
        self._iterIdx = 0
    
    def __len__(self) -> int:
        l = 0
        for s in self._segments:
            l += len(s)
        return l
    
    def __iter__(self):
        return PathIterator(self)
    
    def __getitem__(self, idx: int) -> 'Waypoint':
        s, w = self.index2sw(idx)
        return self._segments[s][w]

    def inverse(self, previousWP: 'Waypoint', solver: 'Screwkincalc'):
        previous = previousWP.jointangles
        for wp in self:
            wp.setRot(previousWP.rot)
            previous = wp.inverse(previous, solver)
        return previous
    
    def newSegment(self) -> 'PathSegment':
        self._segments.append(PathSegment())
        return self._segments[-1]
     
    def index2sw(self, idx: int) -> tuple[int, int]:
        s = 0
        while idx >= len(self._segments[s]):
            idx -= len(self._segments[s])
            s += 1
        return s, idx
    
    def extend(self, wpList: List['Waypoint']) -> None:
        if not self._segments:
            self.newSegment()
        self._segments[-1].Waypoints.extend(wpList)
    
class PathIterator():
    def __init__(self, path: 'Path') -> None:
        self._path = path
        self._idx = 0
    
    def __iter__(self):
        return self
    
    def __next__(self) -> 'Waypoint':
        if self._path is not None and self._idx < len(self._path):
            wp = self._path[self._idx]
            self._idx += 1
            return wp
        else:
            self._path = None
            raise StopIteration


class PathSegment():
    def __init__(self) -> None:
        self._Waypoints: List[Waypoint] = []
    
    def __iter__(self):
        return SegmentIterator(self)
    
    def __getitem__(self, idx: int) -> 'Waypoint':
        return self._Waypoints[idx]
    
    def __len__(self) -> int:
        return len(self._Waypoints)
    
    @property
    def Waypoints(self) -> List[Waypoint]:
        return self._Waypoints
    
    def waypointsXYZ(self):
        x = [wp._cartesian[0] for wp in self._Waypoints]
        y = [wp._cartesian[1] for wp in self._Waypoints]
        z = [wp._cartesian[2] for wp in self._Waypoints]
        return x, y, z
    

class SegmentIterator():
    def __init__(self, segment: 'PathSegment') -> None:
        self._segment = segment
        self._idx = 0
    
    def __iter__(self):
        return self
    
    def __next__(self) -> 'Waypoint':
        if self._segment is not None and self._idx < len(self._segment.Waypoints):
            wp = self._segment.Waypoints[self._idx]
            self._idx += 1
            return wp
        else:
            self._segment = None
            raise StopIteration