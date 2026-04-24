import  imgaug  as  ia
import  imgaug.augmenters  as  iaa
import  numpy  as  np
import  skimage.data  as  skdata
import  cv2

a=5000
b=10

def  c():
  d=skdata.astronaut()
  d=ia.imresize_single_image(d,(128,  128))
  print( "image shape:" , d.shape)
  print( "Press any key or wait %d ms to proceed to the next image."  %  ( a ,))

  e=[
      (1,  75,  75),
      (3,  75,  75),
      (5,  75,  75),
      (10,  75,  75),
      (10,  25,  25),
      (10,  250,  150),
      (15,  75,  75),
      (15,  150,  150),
      (15,  250,  150),
      (20,  75,  75),
      (40,  150,  150),
      ((1,  5),  75,  75),
      (5,  (10,  250),  75),
      (5,  75,  (10,  250)),
      (5,  (10,  250),  (10,  250)),
      (10,  (10,  250),  (10,  250)),
  ]

  cv2.namedWindow( "aug" ,  cv2.WINDOW_NORMAL)
  cv2.resizeWindow( "aug" ,  128*b,  128)

  for  (f,  g,  h)  in  e:
    i=iaa.BilateralBlur(d=f,  sigma_color=g,  sigma_space=h)

    j=[i.augment_image(d)  for  _  in  range(b)]
    j=np.hstack(j)
    print( "dtype" , j.dtype,  "averages" , np.average(j,  axis=tuple(range(0,  j.ndim-1))))

    k="d=%s, sc=%s, ss=%s"  %  (str(f),  str(g),  str(h))
    j=ia.draw_text(j,  x=5,  y=5,  text=k)

    cv2.imshow( "aug" ,  j[...,  ::-1])
    cv2.waitKey(a)


if  __name__  ==  "__main__":
  c()