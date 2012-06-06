import unittest
from dividingline import synthetic_image
from dividingline import DividingLine
import numpy as np


class TestDividingLine(unittest.TestCase):
    def test_dividingline(self):

        line = [1,1,-300]        
        image = synthetic_image((640,480), line)

        d = DividingLine(image)

        result = d.traverse(line)

        assert result['p0'] + result['n0'] == image.shape[0] * image.shape[1]
        assert result['p1'] + result['n1'] == image.sum()
        assert result['p2'] + result['n2'] == (image.astype('i8')**2).sum()

        d2 = DividingLine(image)
        d2.traverse(line)
        assert np.all(d2.debug == d.debug)

        within_eps = lambda a, b: np.all(np.abs(a - b) < 1e-5)
        assert within_eps(float(result['p1'])/result['p0'], 255)
        assert within_eps(float(result['n1'])/result['n0'], 0)

        # TODO: Check that the numpy version matches the cython version


if __name__ == '__main__':
    unittest.main()
