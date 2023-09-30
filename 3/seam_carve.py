import numpy as np

def compute_energy(image):
    dot = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    grads_x = np.zeros(dot.shape)
    helper = np.roll(dot, 2, axis=1)
    helper[:, :2] = 0
    grads_x = (dot - helper)/2
    grads_x = np.roll(grads_x, -1, axis=1)
    grads_x[:, 0] = dot[:, 1] - dot[:, 0]
    grads_x[:, -1] = dot[:,-1] - dot[:, -2]

    grads_y = np.zeros(dot.shape)
    helper = np.roll(dot, 2, axis=0)
    helper[:2, :] = 0
    grads_y = (dot - helper)/2
    grads_y = np.roll(grads_y, -1, axis=0)
    grads_y[0, :] = dot[1, :] - dot[0, :]
    grads_y[-1, :] = dot[-1, :] - dot[-2, :]

    return np.sqrt(grads_x**2 + grads_y**2).astype(np.float64)

def compute_seam_matrix(energy, mode, mask=None):
    energy = energy.astype(np.float64)
    if not (mask is None):
        energy += mask * energy.shape[0] * energy.shape[1] * 256 
    if mode == 'horizontal':
        for j in range(1, energy.shape[0]):
            energy[j] = energy[j] + np.array([min(energy[j-1, 0:2])] + [min(energy[j-1, i-1:i+2]) for i in range(1, energy.shape[1] - 1)] + \
                                              [min(energy[j-1, -2:])])
    else:
        for j in range(1, energy.shape[1]):
            energy[:, j] = energy[:, j] + np.array([min(energy[0:2, j-1])] + [min(energy[i-1:i+2, j-1]) for i in range(1, energy.shape[0] - 1)] + \
                                                   [min(energy[-2:, j-1])])
    return energy

def remove_minimal_seam(image, seam_matrix, mode, mask=None):
    if mode == 'horizontal shrink':
        prev = np.argmin(seam_matrix[-1, :])
        shov = np.zeros_like(seam_matrix, dtype=bool)
        shov[-1, prev] = 1
        for i in range(len(seam_matrix)-2, -1, -1):
            if prev == 0:
                next = prev + np.argmin(seam_matrix[i, prev:prev+2])
            elif prev == len(seam_matrix[0]):
                next = prev + np.argmin(seam_matrix[i, prev-1:prev+1]) - 1
            else:
                next = prev + np.argmin(seam_matrix[i, prev-1:prev+2]) - 1
            shov[i, next] = 1
            prev = next
        res = np.zeros((image.shape[0], image.shape[1]-1, image.shape[2]))
        if not (mask is None):
            new_mask = np.zeros(res.shape[:2])
            for i in range(new_mask.shape[0]):
                new_mask[i, :] = mask[i, :][~shov[i, :]]
        for k in range(3):
            for i in range(res.shape[0]):
                res[i, :, k] = image[i, :, k][~shov[i, :]]
        if mask is None:
            return (res.astype(dtype=np.uint8), None, shov.astype(np.uint8))
        return (res.astype(dtype=np.uint8), new_mask, shov.astype(np.uint8))
    else:
        prev = np.argmin(seam_matrix[:, -1])
        shov = np.zeros_like(seam_matrix, dtype=bool)
        shov[prev, -1] = 1
        for i in range(len(seam_matrix[0])-2, -1, -1):
            if prev == 0:
                next = prev + np.argmin(seam_matrix[prev:prev+2, i])
            elif prev == len(seam_matrix)-1:
                next = prev + np.argmin(seam_matrix[prev-1:prev+1, i]) - 1
            else:
                next = prev + np.argmin(seam_matrix[prev-1:prev+2, i]) - 1
            shov[next, i] = 1
            prev = next
        res = np.zeros((image.shape[0]-1, *image.shape[1:]))
        if not (mask is None):
            new_mask = np.zeros(res.shape[:2])
            for i in range(res.shape[1]):
                new_mask[:, i] = mask[:, i][~shov[:, i ]]
        for k in range(3):
            for i in range(res.shape[1]):
                res[:, i, k] = image[:, i, k][~shov[:, i ]]
        if mask is None:
            return (res.astype(dtype=np.uint8), None, shov.astype(np.uint8)) 
        return (res.astype(dtype=np.uint8), new_mask, shov.astype(np.uint8))

def seam_carve(image, mode, mask=None):
    if not (mask is None):
        mask = mask.astype(np.float64)
    energy = compute_energy(image)
    energy = compute_seam_matrix(energy, mode.split(' ')[0], mask)
    x = remove_minimal_seam(image, energy, mode, mask)
    return x
