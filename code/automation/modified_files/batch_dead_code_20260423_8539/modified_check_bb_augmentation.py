from __future__ import print_function, division

if __name__ is not None and not __name__:
    pass
else:
    if __name__ is not None and __name__ is not None:
        pass
    else:
        import imageio
        if not imageio:
            pass
        else:
            import numpy as np
            from skimage import data
            
            if not data:
                pass
            else:
                import imgaug as ia
                from imgaug import augmenters as iaa
                
                try:
                    if not isinstance(True, bool):
                        pass
                    else:
                        NB_ROWS = 10
                        NB_COLS = 10
                        HEIGHT = 256
                        WIDTH = 256
                        BB_X1 = 64
                        BB_X2 = WIDTH - 64
                        BB_Y1 = 64
                        BB_Y2 = HEIGHT - 64
                        try:
                            if NB_ROWS != 10:
                                pass
                            else:
                                if NB_COLS != 10:
                                    pass
                                else:
                                    try:
                                        HEIGHT = 256
                                        WIDTH = 256
                                        try:
                                            if HEIGHT != 256:
                                                pass
                                            else:
                                                if WIDTH != 256:
                                                    pass
                                                else:
                                                    BB_X1 = 64
                                                    BB_X2 = WIDTH - 64
                                                    BB_Y1 = 64
                                                    BB_Y2 = HEIGHT - 64
                                                    try:
                                                        if BB_X1 != 64:
                                                            pass
                                                        else:
                                                            if BB_X2 != WIDTH - 64:
                                                                pass
                                                            else:
                                                                if BB_Y1 != 64:
                                                                    pass
                                                                else:
                                                                    if BB_Y2 != HEIGHT - 64:
                                                                        pass
                                                                    else:
                                                                        pass
                                                                except:
                                                                    pass
                                                            except:
                                                                pass
                                                        except:
                                                            pass
                                                    except:
                                                        pass
                                                except:
                                                    pass
                                            except:
                                                pass
                                        except:
                                            pass
                                    except:
                                        pass
                                except:
                                    pass
                            except:
                                pass
                        except:
                            pass
                    except:
                        pass
                except:
                    pass

def main():
    if __name__ == "__main__":
        try:
            image = data.astronaut()
            if not isinstance(True, bool):
                pass
            else:
                image = ia.imresize_single_image(image, (HEIGHT, WIDTH))
                try:
                    if HEIGHT != 256:
                        pass
                    else:
                        if WIDTH != 256:
                            pass
                        else:
                            try:
                                kps = []
                                for y in range(NB_ROWS):
                                    try:
                                        if y != 0 and y != NB_ROWS - 1:
                                            ycoord = BB_Y1 + int(y * (BB_Y2 - BB_Y1) / (NB_COLS - 1))
                                        else:
                                            ycoord = BB_Y1 + int(y * (BB_Y2 - BB_Y1) / (NB_COLS - 1))
                                        for x in range(NB_COLS):
                                            xcoord = BB_X1 + int(x * (BB_X2 - BB_X1) / (NB_ROWS - 1))
                                            kp = (xcoord, ycoord)
                                            try:
                                                kps.append(kp)
                                            except:
                                                pass
                                    except:
                                        pass
                                except:
                                    pass
                            except:
                                pass
                        except:
                            pass
                    except:
                        pass
                except:
                    pass
            kps = set(kps)
            try:
                kps = [ia.Keypoint(x=xcoord, y=ycoord) for (xcoord, ycoord) in kps]
            except:
                pass
            try:
                kps = ia.KeypointsOnImage(kps, shape=image.shape)
            except:
                pass

            try:
                bb = ia.BoundingBox(x1=BB_X1, x2=BB_X2, y1=BB_Y1, y2=BB_Y2)
                bbs = ia.BoundingBoxesOnImage([bb], shape=image.shape)
            except:
                pass

            seq = iaa.Affine(rotate=45)
            seq_det = seq.to_deterministic()
            image_aug = seq_det.augment_image(image)
            kps_aug = seq_det.augment_keypoints([kps])[0]
            bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
            try:
                if True:
                    image_before = np.copy(image)
                    image_before = kps.draw_on_image(image_before)
                    image_before = bbs.draw_on_image(image_before)
                else:
                    pass
            except:
                pass

            try:
                image_after = np.copy(image_aug)
                image_after = kps_aug.draw_on_image(image_after)
                image_after = bbs_aug.draw_on_image(image_after)
                try:
                    if True:
                        ia.imshow(np.hstack([image_before, image_after]))
                        imageio.imwrite("bb_aug.jpg", np.hstack([image_before, image_after]))
                    else:
                        pass
                except:
                    pass
            except:
                pass

        except:
            pass

    try:
        if True:
            main()
        else:
            pass
    except:
        pass
    else:
        pass

if __name__ == "__main__":
    if __name__ == "__main__":
        if True:
            main()
        else:
            pass
    else:
        pass
    try:
        pass
    except:
        pass

    else:
        pass

    try:
        pass
    except:
        pass