import math
import random
import cv2
import skimage.transform
import numpy as np


def change_contrast(img, alpha_range=(0.0, 2.0)):
    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    alpha = random.uniform(alpha_range[0], alpha_range[1])
    img2[:,:,0] = cv2.multiply(img[:,:,0], np.array([alpha]))
    img2 = cv2.cvtColor(img2, cv2.COLOR_Lab2RGB)
    return img2


def randint_same(start, end):
    """If asked to generated a random number between start and end where
    start >= end then return start."""
    if start >= end:
        return start
    return random.randint(start, end)


def draw_shadow(img, alpha_min=0.7, alpha_max=1.0):
    img2 = np.copy(img)

    rows, cols = img.shape[:2]
    kind = random.choice(['left', 'bottom', 'right', 'top'])
    if kind == 'left':
        pt_first = [0, randint_same(0, rows-2)]
        pt_last = [0, randint_same(pt_first[1], rows)]
        pt1 = [randint_same(0, cols), randint_same(pt_first[1], pt_last[1]-1)]
        pt2 = [randint_same(0, cols), randint_same(pt1[1], pt_last[1])]
    elif kind == 'bottom':
        pt_first = [randint_same(0, cols-1), rows]
        pt_last = [randint_same(pt_first[0], cols), rows]
        pt1 = [randint_same(pt_first[0], cols), randint_same(0, rows)]
        pt2 = [randint_same(pt1[0], cols), randint_same(0, rows)]
    elif kind == 'right':
        pt_first = [cols, randint_same(0, rows-2)]
        pt_last = [cols, randint_same(pt_first[1], rows)]
        pt1 = [randint_same(0, cols), randint_same(pt_first[1], pt_last[1]-1)]
        pt2 = [randint_same(0, cols), randint_same(pt1[1], pt_last[1])]
    elif kind == 'top':
        pt_first = [randint_same(0, cols-1), 0]
        pt_last = [randint_same(pt_first[0], cols), 0]
        pt1 = [randint_same(pt_first[0], cols), randint_same(0, rows)]
        pt2 = [randint_same(pt1[0], cols), randint_same(0, rows)]
    pts = np.array([
        pt_first,
        pt1,
        pt2,
        pt_last,
    ], np.int32)
    overlay = cv2.fillPoly(img2, pts=[pts], color=(0, 0, 0))
    alpha = random.uniform(alpha_min, alpha_max)
    img_new = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img2)
    return img_new


def translation(img, x_delta=30, y_delta=30):
    t_x = random.randint(-x_delta, x_delta)
    t_y = random.randint(-y_delta, y_delta)
    M = np.float32([
        [1, 0, t_x],
        [0, 1, t_y],
    ])
    return (cv2.warpAffine(img, M, img.shape[:2]), t_x)


def shear(img,
          center_min=0.9,
          center_max=1.1):
    rows, cols = img.shape[:2]
    x_center = random.uniform(math.floor(cols/2*center_min), math.ceil(cols/2*center_max))
    y_center = random.uniform(math.floor(rows/2*center_min), math.ceil(rows/2*center_max))
    pts1 = np.float32([
        [0, rows],            # bottom left
        [cols, rows],         # bottom right
        [cols / 2, rows / 2], # midpoint
    ])
    pts2 = np.float32([
        [0, rows],            # bottom left
        [cols, rows],         # bottom right
        [x_center, y_center], # random point
    ])
    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(img, M, (cols, rows),
                          borderMode=cv2.BORDER_REPLICATE)


def augment(img,
            new_image_count=10,
            with_contrast=True,
            with_shadow=True,
            with_shear=False):
    """Shadow deliberately comes after contrast because on the first track
    just before the bridge the road is very bright but the tree shadow on
    the left is very dark.
    """
    output = [np.copy(img)]
    for i in range(new_image_count):
        new_img = np.copy(img)
        if with_contrast:
            new_img = change_contrast(new_img)
        if with_shadow:
            new_img = draw_shadow(new_img)
        if with_shear:
            new_img = shear(new_img)
        output.append(new_img)
    return output


def augment_batch(batch_x,
                  batch_y,
                  new_image_count=10,
                  with_contrast=True,
                  with_shadow=True,
                  with_shear=False):
    output = []
    for (x, y) in zip(batch_x, batch_y):
        for new_img in augment(x,
                               new_image_count,
                               with_contrast,
                               with_shadow,
                               with_shear):
            output.append((new_img, y))
    x, y = map(np.array, list(zip(*output)))
    return (x, y)


def preprocess_img(img, crop_top, crop_bottom=0):
    # Convert the color space to L*a*b*, where the first channel contains
    # most of the perceptual information we care about.
    img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)

    # Apply equalization to different tiles of the Y channel that
    # contains most of the information. This improves the consistency
    # and contrast of images. But makes driving much worse on track 2.
    #rows, cols = img.shape[:2]
    #tileGridSize = (int(rows/15), int(cols/15))
    #clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=tileGridSize)
    #img[:,:,0] = clahe.apply(img[:,:,0])

    # Crop the top crop_top and bottom crop_bottom pixels from the image.
    rows, cols = img.shape[:2]
    img = img[crop_top:rows-crop_bottom, :]

    # Resize the image to reduce training time, without affecting
    # accuracy too much
    img = cv2.resize(img, (128, 128), cv2.INTER_LANCZOS4)

    # Normalize the range of values to reduce the time taken for
    # gradient descent style algorithms to converge.
    img = (img.astype(np.float32) / 255.0) - 0.5

    return img
