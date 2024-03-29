import dividingline; reload(dividingline)
from dividingline import dividingline
reload(dividingline)
from dividingline import DividingLine, synthetic_image, random_middle_line
import pylab
import scipy.optimize

line = [1,1,-300]
#im = synthetic_image((640,480), line)
im = imread('tableshot.png').mean(2)[100:,:]*255
d = DividingLine(im)
res = d.traverse(line, True)

d2 = DividingLine(im)
res2 = d2.traverse_np(line, True)


def animate():
    while True:

        line = random_middle_line()
        d = DividingLine(synthetic_image(line=line))
        d.traverse(line, True)
        #d.traverse_np(line, True)
        pylab.clf()
        pylab.imshow(d.debug)
        pylab.waitforbuttonpress(0.01)


middle_offset = dividingline.middle_offset
def optimize():
    size = im.shape[::-1]
    def error(x):
        theta, dist = x
        line = middle_offset(theta, dist, size)
        s =  1./(d.score(line, True) + 1e-5)
        clf()
        imshow(d.debug * d.image)
        pylab.waitforbuttonpress(0.01)
        return s

    initial = (np.random.rand()*2*pi, (2*np.random.rand()-1)*100)
    r = scipy.optimize.fmin(error, initial)

    line = middle_offset(r[0], r[1], size)
    line = np.array(line) / line[2]
    res = d.traverse(line)

    # Scale the line so that a positive dot product indicates that a point is
    # on the 'bright' side of the line
    if res['p1'] / res['p0'] < res['n1'] / res['n0']:
        line *= -1

    return line
