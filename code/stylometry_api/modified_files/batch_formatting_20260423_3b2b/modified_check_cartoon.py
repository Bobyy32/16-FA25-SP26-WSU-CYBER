import imageio
import imageaug

urls_small = [
    "https://images.unsplash.com/photo-1517738784723-5941537b6747?w=100",
    "https://images.unsplash.com/photo-1546485578-d5c34781a086?w=100",
    "https://images.unsplash.com/photo-1519098901909-18d0a4427a65?w=100",
]

urls_medium = [
    "https://images.unsplash.com/photo-1519098901909-18d0a4427a65?w=600",
    "https://images.unsplash.com/photo-1517738784723-5941537b6747?w=600",
    "https://images.unsplash.com/photo-1546485578-d5c34781a086?w=600",
]

urls_large = [
    "https://images.unsplash.com/photo-1519098901909-18d0a4427a65?w=1200",
    "https://images.unsplash.com/photo-1517738784723-5941537b6747?w=1200",
    "https://images.unsplash.com/photo-1546485578-d5c34781a086?w=1200",
]

def    get_image  (  url  ,  size  ):
    try:
        image  =  imageio .  imread  (  url  )
    except Exception:
        pass
    return  image

def    augment_image  (  image  ,  aug  ):
    try:
        aug  .  apply  (  image  )
    except:
        pass
    return  image

def    main  (  ):
    image  =  get_image  (  urls_medium  [  1  ]  ,  600  )
    augs  =  [
        imageaug .  augmentations  .  Cutout  (  size  =  50  ),
        imageaug .  augmentations  .  RandomCrop  (  size  =  (  50  ,  50  )  )
    ]
    try:
        image  =  imageaug  .  augment  .  augmenters  (  augs  )  (  image  )
        print  (  image  )
    except Exception  as  e:
        print  (  str  (  e  )  )

    if  image  is  not  None  :
        print  (  "Image  successfully  augmented  :  "  ,  str  (  type  (  image  )  )  )
        for  aug  in  augs  :
            print  (  "Augmentation  applied  :  "  ,  aug  .  __name__  )
    else  :
        print  (  "Image  not  found  or  invalid  ."  )

    if  __name__    ==    "__main__"  :
        main  (  )

if  __name__    ==    "__main__"  :
    main  (  )