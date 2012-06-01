"""
Andrew Miller   May 2012

Collect all the moments from an integral image on either side of a line
"""

cimport numpy as np
import numpy as np
import cython

cdef struct Result:
    np.uint32_t p0
    np.uint32_t p1
    np.uint64_t p2

    np.uint32_t n0
    np.uint32_t n1
    np.uint64_t n2


cdef Result _traverse_moments(np.uint32_t *m1,
                              np.uint64_t *m2,
                              int width, int height,
                              int l, int t, int r, int b,
                              float A, float B, float C,
                              dbg=None):

    cdef Result result
    result.p0 = result.p1 = result.p2 = 0
    result.n0 = result.n1 = result.n1 = 0

    cdef np.uint32_t r0, r1
    cdef np.uint64_t r2
    cdef int p0, p1, p2, p3
    cdef Result lt, lb, rt, rb
    cdef int x, y

    if r-l < 1 or b-t < 1:
        return result

    p0 =     l*A +     t*B + C > 0
    p1 =     l*A + (b-1)*B + C > 0
    p2 = (r-1)*A +     t*B + C > 0
    p3 = (r-1)*A + (b-1)*B + C > 0

    if (p0 == p1 == p2 == p3):
        # We can stop traversing here - entirely on one side of the line

        if dbg is not None: dbg[t+1:b,l+1:r] = p0*2-1

        r0 = (r-l)*(b-t)
        r1 = m1[t*width+l] + m1[b*width+r] - m1[b*width+l] - m1[t*width+r]
        r2 = m2[t*width+l] + m2[b*width+r] - m2[b*width+l] - m2[t*width+r]

        if p0: result.p0 = r0; result.p1 = r1; result.p2 = r2
        else:  result.n0 = r0; result.n1 = r1; result.n2 = r2

        return result

    else:
        # Recursively call the four sub-quadrants

        x = (l+r)/2
        y = (t+b)/2

        lt = _traverse_moments(m1, m2, width, height, l, t, x, y, A, B, C, dbg)
        rt = _traverse_moments(m1, m2, width, height, x, t, r, y, A, B, C, dbg)
        lb = _traverse_moments(m1, m2, width, height, x, y, r, b, A, B, C, dbg)
        rb = _traverse_moments(m1, m2, width, height, l, y, x, b, A, B, C, dbg)

        result.p0 = lt.p0 + rt.p0 + lb.p0 + rb.p0
        result.p1 = lt.p1 + rt.p1 + lb.p1 + rb.p1
        result.p2 = lt.p2 + rt.p2 + lb.p2 + rb.p2
        result.n0 = lt.n0 + rt.n0 + lb.n0 + rb.n0
        result.n1 = lt.n1 + rt.n1 + lb.n1 + rb.n1
        result.n2 = lt.n2 + rt.n2 + lb.n2 + rb.n2

        return result


def traverse_moments(np.ndarray[np.uint32_t, ndim=2, mode='c'] m1,
                     np.ndarray[np.uint64_t, ndim=2, mode='c'] m2,
                     line, dbg=None):

    height, width = m1.shape[0], m1.shape[1]
    (A,B,C) = line
    
    return _traverse_moments(<np.uint32_t *> m1.data,
                             <np.uint64_t *> m2.data,
                             width, height,
                             0, 0, width-1, height-1, A, B, C, dbg)

