"""
Andrew Miller   May 2012


"""

import numpy as np
import operator
from dividingline_cy import traverse_moments


class DividingLine(object):
    def __init__(self, image, weights=None):
        assert image.dtype in (np.uint8, np.float32)
        assert len(image.shape) == 2, "Greyscale only for now"

        def int_image(im):
            i = np.zeros((im.shape[0]+1, im.shape[1]+1), im.dtype)
            i[1:,1:] = im
            return np.cumsum(np.cumsum(i, 0), 1)

        self.image = image

        # Compute the integral image
        self.m1 = int_image(image.astype('u4'))
        self.m2 = int_image(image.astype('u8')**2)


    def traverse_np(self, line, debug=False):
        self.debug = None if not debug else np.zeros_like(self.image).astype('f')
        rect = ((0,0),self.image.shape[::-1])
        f = self._getmoments
        def g((pa, na), (pb, nb)): 
            return map(sum, zip(pa,pb)), map(sum, zip(na,nb))
        #g = lambda (pa, na), (pb, nb): map(sum, zip(pa,pb)), map(sum, zip(na,nb))
        return self._traverse(line, rect, f, g, debug=self.debug)

    def traverse(self, line, debug=False):
        self.debug = None if not debug else np.zeros_like(self.image).astype('f')
        return traverse_moments(self.m1, self.m2, line, self.debug)

    def _getrect_plain(self, im, ((l,t),(r,b))):
        # Same as _getrect, but without an integral image
        return im[t:b,l:r].sum()

    def _getrect(self, im, ((l,t),(r,b))):
        # im must be an integral image
        return im[t,l] + im[b,r] - im[t,r] - im[b,l]
    
    def _getmoments(self, rect, pn):
        # All three moments
        ((l,t),(r,b)) = rect
        m0 = (r-l)*(b-t)
        m1 = self._getrect(self.m1, rect)
        m2 = self._getrect(self.m2, rect)

        if pn: return (m0, m1, m2), (0, 0, 0)
        else:  return (0, 0, 0), (m0, m1, m2)

    def _traverse(self, line, rect, 
                  f = lambda _: 0,
                  g = operator.add, # addition-like
                  debug=None):      # debug output image

        (l,t),(r,b) = rect

        points = np.array([[  l,t,1], [  l,b-1,1],
                           [r-1,t,1], [r-1,b-1,1]])
        p = np.dot(points, line) > 0

        if p[0] == p[1] == p[2] == p[3] or r-l < 1 or b-t < 1 or \
                (r-l <= 1 and b-t <= 1):

            if debug is not None: debug[t+1:b,l+1:r] = p[0]*2-1

            # We apply the base case when either:
            # a: the entire rectangle is on one side
            #    of the line, or
            # b: there's only one pixel
            return f(rect, p[0])
        else:

            # Find the center point...
            x = (l+r)/2
            y = (t+b)/2

            # ... then split into four sub-rectangles
            subs = [((l,t), (x,y)),
                    ((x,t), (r,y)),
                    ((x,y), (r,b)),
                    ((l,y), (x,b))]

            # Apply apply g to the four sub problems
            res = self._traverse(line, subs[0], f, g, debug)
            for s in subs[1:]:
                res = g(res, self._traverse(line, s, f, g, debug))
            return res


    def score(self, line, debug=False):
        # Bhattacharyya coefficient
        r = self.traverse(line, debug)
        if not r['p0'] or not r['n0']: return 0

        u1 = float(r['p1']) / r['p0']
        v1 = float(r['p2']) / r['p0'] - u1**2
        
        u2 = float(r['n1']) / r['n0']
        v2 = float(r['n2']) / r['n0'] - u2**2

        return (u1-u2) ** 2 / (v1 + v2 + 1e-5)


def random_middle_line(size=(640,480)):
    # Returns a line passing through the center of the image
    # at a random orientation
    a,b = np.random.rand(2)-0.5
    c = -(size[0]/2*a + size[1]/2*b)
    return np.array([a,b,c], 'f')


def synthetic_image(size, line=[1,1,-300]):
    x,y = np.meshgrid(range(size[0]), range(size[1]))
    w = np.ones_like(x)

    b = np.sum(line * np.dstack((x,y,w)), axis=2)
    return (b > 0).astype('u1') * 255


def middle_offset(theta, d, size=(640,480)):
    a,b = -np.sin(theta), np.cos(theta)
    x = a * d + size[0]/2
    y = b * d + size[1]/2
    c = -(x * a + y * b)
    return a,b,c
        
    
def optimize(d):
    pass
