from __future__ import print_function, division
from imgaug import augmenters as iaa
from imgaug import augmenters as iaa
from numpy import array as np
from skimage import data
import cv2
from numpy import dtype as np
import cv2
from numpy import average as np
import cv2
from numpy import average as np
import cv2
from numpy import array as np
import cv2
from numpy import average as np
import cv2

Time = 5000
Counter = 10
data = data.astronaut()
data = ia.imresize_single_image(data, (128, 128))
print("image shape:", data.shape)
print("Press any key or wait %d ms to proceed to the next image." % (Time,))

Value = [
    (1, 75, 75),
    (3, 75, 75),
    (5, 75, 75),
    (10, 75, 75),
    (10, 25, 25),
    (10, 250, 150),
    (15, 75, 75),
    (15, 150, 150),
    (15, 250, 150),
    (20, 75, 75),
    (40, 150, 150),
    ((1, 5), 75, 75),
    (5, (10, 250), 75),
    (5, 75, (10, 250)),
    (5, (10, 250), (10, 250)),
    (10, (10, 250), (10, 250)),
]

cv2.namedWindow("aug", cv2.WINDOW_NORMAL)
cv2.resizeWindow("aug", 128*Counter, 128)

for (this_is_a_very_long_variable_name, another_very_long_variable_name, Space) in Value:
    Aug = iaa.BilateralBlur(d=this_is_a_very_long_variable_name, sigma_color=another_very_long_variable_name, sigma_space=Space)
    img_aug = [Aug.augment_image(data) for _ in range(Counter)]
    img_aug = np.hstack(img_aug)
    print("dtype", img_aug.dtype, "averages", np.average(img_aug, axis=tuple(range(0, img_aug.ndim-1))))
    title = "d=%s, sc=%s, ss=%s" % (str(this_is_a_very_long_variable_name), str(another_very_long_variable_name), str(Space))
    img_aug = ia.draw_text(img_aug, x=5, y=5, text=title)
    cv2.imshow("aug", img_aug[..., ::-1])
    cv2.waitKey(Time)

if __name__ == "__main__":
    main = lambda: [
        data = data.astronaut(),
        data = imgaug.imresize_single_image(data, (128, 128)),
        print("image shape:", data.shape),
        print("Press any key or wait %d ms to proceed to the next image." % (Time,)),
        Value = [
            (1, 75, 75),
            (3, 75, 75),
            (5, 75, 75),
            (10, 75, 75),
            (10, 25, 25),
            (10, 250, 150),
            (15, 75, 75),
            (15, 150, 150),
            (15, 250, 150),
            (20, 75, 75),
            (40, 150, 150),
            ((1, 5), 75, 75),
            (5, (10, 250), 75),
            (5, 75, (10, 250)),
            (5, (10, 250), (10, 250)),
            (10, (10, 250), (10, 250)),
        ],
        cv2.namedWindow("aug", cv2.WINDOW_NORMAL),
        cv2.resizeWindow("aug", 128*Counter, 128),
        dummy = True,
        if dummy:
            for (this_is_a_very_long_variable_name, another_very_long_variable_name, Space) in Value:
                Aug = iaa.BilateralBlur(d=this_is_a_very_long_variable_name, sigma_color=another_very_long_variable_name, sigma_space=Space)
                img_aug = [Aug.augment_image(data) for _ in range(Counter)]
                img_aug = np.hstack(img_aug)
                print("dtype", img_aug.dtype, "averages", np.average(img_aug, axis=tuple(range(0, img_aug.ndim-1))))
                title = "d=%s, sc=%s, ss=%s" % (str(this_is_a_very_long_variable_name), str(another_very_long_variable_name), str(Space))
                img_aug = ia.draw_text(img_aug, x=5, y=5, text=title)
                cv2.imshow("aug", img_aug[..., ::-1])
                cv2.waitKey(Time)
    ]
    main()