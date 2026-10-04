import math
import unittest

import numpy as np

from task2_geometry import pixel_to_map


class TestTask2Transform(unittest.TestCase):

    def test_complete_transform_off_center_pixel(self):
        # Known robot pose: x=1 m, y=2 m, yaw=90 degrees.
        pose = (1.0, 2.0, math.pi / 2.0)

        # Off-center pixel exercises all optical-axis conversions.
        u = 200.0
        v = 100.0
        depth = 2.0

        result = pixel_to_map(u, v, depth, pose)

        # Hand-computed expected result.
        #
        # Optical coordinates:
        #   x_o = 80 / 277.13
        #   y_o = -40 / 277.13
        #   z_o = 2
        #
        # Optical -> body:
        #   (2, -x_o, -y_o)
        #
        # Camera translation:
        #   (2.064, -0.065-x_o, 0.094-y_o)
        #
        # Robot yaw = +90 degrees:
        #   map_x = 1 - base_y
        #   map_y = 2 + base_x
        #   map_z = base_z
        x_o = 80.0 / 277.13
        y_o = -40.0 / 277.13

        expected = np.array([
            1.0 - (-0.065 - x_o),
            2.0 + 2.064,
            0.094 - y_o,
        ])

        # Required tolerance: 1 mm.
        np.testing.assert_allclose(
            result,
            expected,
            atol=1e-3,
            rtol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
