import cv2
import os
import glob

import fusion


def cv_show(image, message):
    cv2.imshow(message, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    IMAGES_DIR = './pics'
    SAVE_DIR = './save'

    images_list = glob.glob(os.path.join(IMAGES_DIR, '*'))
    images = [cv2.imread(e) for e in images_list]

    naive_fusion = fusion.naive_fusion(images)
    gaussian_fusion = fusion.gaussian_fusion(images)
    laplacian_fusion = fusion.laplacian_fusion(images)

    cv_show(naive_fusion, 'naive_fusion')
    cv_show(gaussian_fusion, 'gaussian_fusion')
    cv_show(laplacian_fusion, 'laplacian_fusion')


    os.makedirs(SAVE_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(SAVE_DIR, 'naive_fusion.jpg'), naive_fusion)
    cv2.imwrite(os.path.join(SAVE_DIR, 'gaussian_fusion.jpg'), gaussian_fusion)
    cv2.imwrite(os.path.join(SAVE_DIR, 'laplacian_fusion.jpg'), laplacian_fusion)

