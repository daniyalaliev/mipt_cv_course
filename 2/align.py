import numpy as np

def align(img, coords):
    img = img[img.shape[0]%3:, ...]
    cut = img.shape[0]//3
    work = np.stack((img[:cut, ...], img[cut:2*cut, ...], img[2*cut:, ...]), axis=-1)
    work = work[int(work.shape[0]/10):int(9/10*work.shape[0]), int(work.shape[1]/10):int(9/10*work.shape[1]), ...]

    idx = np.fft.ifft2(np.fft.fft2(work[..., 1]) * np.conjugate(np.fft.fft2(work[..., 0])))
    idx = np.unravel_index(np.argmax(np.abs(idx)), idx.shape)
    first = coords[0] -  cut - idx[0] if idx[0] < work.shape[0]//2 else coords[0] + (work.shape[0] - idx[0]) - cut
    second = coords[1] - idx[1] if idx[1] < work.shape[1]//2 else coords[1] + (work.shape[1]-idx[1])
    blue = (first, second)
    work[..., 0] = np.roll(work[..., 0], idx[0], axis=0)
    work[..., 0] = np.roll(work[..., 0], idx[1], axis=1)

    idx = np.fft.ifft2(np.fft.fft2(work[..., 1]) * np.conjugate(np.fft.fft2(work[..., 2])))
    idx = np.unravel_index(np.argmax(np.abs(idx)), idx.shape)
    first = coords[0] +  cut - idx[0] if idx[0] < work.shape[0]//2 else coords[0] + (work.shape[0] - idx[0]) + cut
    second = coords[1] - idx[1] if idx[1] < work.shape[1]//2 else coords[1] + (work.shape[1]-idx[1])
    red = (first, second)
    work[..., 2] = np.roll(work[..., 2], idx[0], axis=0)
    work[..., 2] = np.roll(work[..., 2], idx[1], axis=1)
    return work[...,::-1], blue, red
