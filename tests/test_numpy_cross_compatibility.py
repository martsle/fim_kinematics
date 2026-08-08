from unittest.mock import patch

import numpy as np

from fimkin import Screwkincalc


def test_inverse_uses_only_supported_cross_product_dimensions():
    original_cross = np.cross

    def reject_two_dimensional_cross(a, b, *args, **kwargs):
        if np.asarray(a).shape == (2,) and np.asarray(b).shape == (2,):
            raise ValueError("two-dimensional vectors are not supported")
        return original_cross(a, b, *args, **kwargs)

    joints = np.array([0.3, -1.2, 1.0, -0.6, -1.4, 0.5])

    with patch.object(np, "cross", side_effect=reject_two_dimensional_cross):
        for robot in (1, 2):
            solver = Screwkincalc(robot=robot, threshold=0.1)
            solver.setDesiredPOE(solver.forward(joints))

            solved = np.asarray(solver.inverse(joints))

            np.testing.assert_allclose(solved, joints, atol=1e-9)
