import cv2
import numpy as np

def feather_blending(image,background,mask,edge=15):
    image16 = np.array(image, dtype=np.int16)
    mask8 = mask
    background16 = np.array(background, dtype=np.int16)
    kernel = np.ones((edge, edge), np.uint8)
    background_mask = 255 - cv2.erode(mask8, kernel, iterations=1)
    fb = cv2.detail_FeatherBlender(sharpness=1./edge)
    # corners = [[0, 0], [1080, 0], [1080, 1920], [0, 1920]]
    corners = fb.prepare((0, 0, image.shape[1], image.shape[0]))
    fb.feed(image16, mask8, corners)
    fb.feed(background16, background_mask, corners)
    output = None
    output_mask = None
    output, output_mask = fb.blend(output, output_mask)
    return output

