from __future__ import print_function, division
import imageio as i
import numpy as n
from skimage import data as d
import imgaug as ia
from imgaug import augmenters as a
x=10
y=10
z=256
w=256
a1=64
a2=w-64
a3=64
a4=z-64
def main():
    b=d.astronaut()
    b=ia.imresize_single_image(b,(z,w))
    c=[]
    for e in range(x):
        f=a3+int(e*(a4-a3)/(y-1))
        for g in range(y):
            h=a1+int(g*(a2-a1)/(x-1))
            i=(h,f)
            c.append(i)
    c=set(c)
    c=[ia.Keypoint(x=h,y=f) for (h,f) in c]
    c=ia.KeypointsOnImage(c,shape=b.shape)
    j=ia.BoundingBox(x1=a1,x2=a2,y1=a3,y2=a4)
    k=ia.BoundingBoxesOnImage([j],shape=b.shape)
    l=a.Affine(rotate=45)
    m=l.to_deterministic()
    b=m.augment_image(b)
    c=m.augment_keypoints([c])[0]
    k=m.augment_bounding_boxes([k])[0]
    o=n.copy(b)
    o=c.draw_on_image(o)
    o=k.draw_on_image(o)
    p=n.copy(b)
    p=c.draw_on_image(p)
    p=k.draw_on_image(p)
    ia.imshow(n.hstack([o,p]))
    i.imwrite("bb_aug.jpg",n.hstack([o,p]))
if __name__=="__main__":
    main()