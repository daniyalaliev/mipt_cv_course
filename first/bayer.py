import numpy as np

def GatR(image, i, j):
    res = np.clip((4 * image[i, j, 0] -1 * (image[i-2, j, 0] + image[i+2, j, 0] + image[i, j+2, 0] + image[i, j-2, 0]) \
        + 2 * (image[i, j-1, 1] + image[i-1, j, 1] + image[i, j+1, 1] + image[i+1, j, 1]))/8, 0, 255)
    return res

def GatB(image, i, j):
    res = np.clip((4 * image[i, j, 2] -1 * (image[i-2, j, 2] + image[i+2, j, 2] + image[i, j+2, 2] + image[i, j-2, 2]) \
        + 2 * (image[i, j-1, 1] + image[i-1, j, 1] + image[i, j+1, 1] + image[i+1, j, 1]))/8, 0, 255)
    return res

def RatG(image, i, j, mask):
    if not mask[i, j-1, 0]:
        res = np.clip((5 * image[i, j, 1] + 4 * (image[i-1, j, 0] + image[i + 1, j, 0]) \
            -1 * (image[i-1, j-1, 1] + image[i+1, j-1, 1] + image[i+1, j+1, 1] + image[i-1, j+1, 1] + image[i-2, j, 1] + image[i+2, j, 1]) \
                +0.5 * (image[i, j-2, 1] + image[i, j+2, 1]))/8, 0, 255)
    else:
        res = np.clip((5 * image[i, j, 1] + 4 * ( image[i, j-1, 0] + image[i, j + 1, 0]) \
            -1 * (image[i-1, j-1, 1] + image[i+1, j-1, 1] + image[i+1, j+1, 1] + image[i-1, j+1, 1] + image[i, j-2, 1] + image[i, j+2, 1]) \
                +0.5 * (image[i+2, j, 1] + image[i-2, j, 1]))/8, 0, 255)
    return res

def RatB(image, i, j):
    res = np.clip((6 * image[i, j, 2] -1.5 * (image[i-2, j, 2] + image[i, j-2, 2] + image[i, j+2, 2] + image[i+2, j, 2]) \
        + 2 * (image[i-1, j-1, 0] + image[i+1, j-1, 0] + image[i+1, j+1, 0] + image[i-1, j+1, 0]))/8, 0, 255)
    return res

def BatG(image, i, j, mask):
    if not mask[i-1, j, 2]:
        res = np.clip((5 * image[i, j, 1] + 4 * (image[i, j+1, 2] + image[i, j-1, 2]) \
            -1 * (image[i-1, j-1, 1] + image[i+1, j-1, 1] + image[i+1, j+1, 1] + image[i-1, j+1, 1]) -1 * (image[i, j-2, 1] + image[i, j+2, 1]) \
            + 0.5 * (image[i-2, j, 1] + image[i+2, j, 1]))/8, 0, 255)
    else:
        res = np.clip((5 * image[i, j, 1] + 4 * (image[i-1, j, 2] + image[i + 1, j, 2]) \
            -1 * (image[i-1, j-1, 1] + image[i+1, j-1, 1] + image[i+1, j+1, 1] + image[i-1, j+1, 1]) + 0.5 * (image[i, j-2, 1] + image[i, j+2, 1]) \
            -1 * (image[i-2, j, 1] + image[i+2, j, 1]))/8, 0, 255)    
    return res

def BatR(image, i, j):
    res = np.clip((6 * image[i, j, 0] -1.5 * (image[i-2, j, 0] + image[i, j-2, 0] + image[i, j+2, 0] + image[i+2, j, 0]) \
        + 2 * (image[i-1, j-1, 2] + image[i+1, j-1, 2] + image[i+1, j+1, 2] + image[i-1, j+1, 2] ))/8, 0, 255)
    return res

def get_bayer_masks(n_rows, n_cols):
    res = np.zeros((n_rows, n_cols, 3), dtype=bool)
    red = np.zeros((n_rows, n_cols))
    red[::2, :] = np.hstack((np.tile([0, 1], n_cols//2), np.zeros(n_cols%2)))
    res[..., 0] = red
    green = np.zeros((n_rows, n_cols))
    green[::2, :] = np.hstack((np.tile([1, 0], n_cols//2), np.ones(n_cols%2)))
    green[1::2, :] = np.hstack((np.tile([0, 1], n_cols//2), np.zeros(n_cols%2)))
    res[..., 1] = green
    blue = np.zeros((n_rows, n_cols))
    blue[1::2, :] = np.hstack((np.tile([1, 0], n_cols//2), np.ones(n_cols%2)))
    res[..., 2] = blue
    return res

def get_colored_img(raw_img):
    res = np.zeros((*raw_img.shape, 3), dtype=np.uint8)
    mask = get_bayer_masks(raw_img.shape[0], raw_img.shape[1])
    res[mask[..., 0], 0] = raw_img[mask[..., 0]]
    res[mask[..., 1], 1] = raw_img[mask[..., 1]]
    res[mask[..., 2], 2] = raw_img[mask[..., 2]]
    return res

def bilinear_interpolation(colored_img):
    res = np.zeros(colored_img.shape, dtype=np.uint8)
    mask = get_bayer_masks(colored_img.shape[0], colored_img.shape[1])
    for k in range(3):
        for i in range(1, colored_img.shape[0] - 1):
            for j in range(1, colored_img.shape[1] - 1):
                if not mask[i, j, k]:
                    res[i, j, k] = np.sum(colored_img[i-1:i+2, j-1:j+2, k])/np.sum(mask[i-1:i+2, j-1:j+2, k])
                else:
                    res[i, j, k] = colored_img[i, j, k]
    return res

def improved_interpolation(raw_img):
    res = np.zeros((*raw_img.shape, 3), dtype=np.uint64)
    mask = get_bayer_masks(raw_img.shape[0], raw_img.shape[1])
    raw_img = get_colored_img(raw_img).astype(np.uint64)
    for k in range(3):
        for i in range(2, raw_img.shape[0] - 2):
            for j in range(2, raw_img.shape[1]-2):
                if mask[i, j, k] == 0:
                    if k == 0:
                        if mask[i, j, 1]:
                            res[i, j, k] = RatG(raw_img, i, j, mask)
                        elif mask[i, j, 2]:
                            res[i, j, k] = RatB(raw_img, i, j)
                    elif k == 1:
                        if mask[i, j, 0]:
                            res[i, j, k] = GatR(raw_img, i, j)
                        elif mask[i, j, 2]:
                            res[i, j, k] = GatB(raw_img, i, j)
                    else:
                        if mask[i, j, 0]:
                            res[i, j, k] = BatR(raw_img, i, j)
                        elif mask[i, j , 1]:
                            res[i, j, k] = BatG(raw_img, i, j, mask)
                else:
                    res[i, j, k] = raw_img[i, j, k]
    return np.clip(res, 0, 255).astype(np.uint8)
            