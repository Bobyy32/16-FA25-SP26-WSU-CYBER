import  __future__
from  imgaug  import  *
import  numpy
from  skimage  import  data
import  cv2

a = 5000
b = 10


def  main (  ):
    c = data.astronaut (  )
    c = ia.imresize_single_image ( c,  ( 128,  128 )  )
    print ( "image shape:",  c.shape )
    print ( "Press any key or wait %d ms to proceed to the next image."  %  ( a, )  )

    d = [
        ( 1,  75,  75 ),
        ( 3,  75,  75 ),
        ( 5,  75,  75 ),
        ( 10,  75,  75 ),
        ( 10,  25,  25 ),
        ( 10,  250,  150 ),
        ( 15,  75,  75 ),
        ( 15,  150,  150 ),
        ( 15,  250,  150 ),
        ( 20,  75,  75 ),
        ( 40,  150,  150 ),
        ( ( 1,  5 ),  75,  75 ),
        ( 5,  ( 10,  250 ),  75 ),
        ( 5,  75,  ( 10,  250 )  ),
        ( 5,  ( 10,  250 ),  ( 10,  250 )  ),
        ( 10,  ( 10,  250 ),  ( 10,  250 )  ),
    ]

    cv2.namedWindow ( "aug",  cv2.WINDOW_NORMAL )
    cv2.resizeWindow ( "aug",  128*b,  128 )

    for  ( e,  f,  g )  in  d:
        h = iaa.BilateralBlur ( d=e,  sigma_color=f,  sigma_space=g )

        i = [ h.augment_image ( c )  for  _  in  range ( b ) ]
        i = numpy.hstack ( i )
        print ( "dtype",  i.dtype,  "averages",  numpy.average ( i,  axis=tuple ( range ( 0,  i.ndim-1 ) ) ) )

        j = "d=%s, sc=%s, ss=%s"  %  ( str ( e ),  str ( f ),  str ( g ) )
        i = ia.draw_text ( i,  x=5,  y=5,  text=j )

        cv2.imshow ( "aug",  i[ ...,  ::-1 ] )
        cv2.waitKey ( a )


if  __name__  ==  "__main__":
    main (  )