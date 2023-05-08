import numpy as np
import cv2


def __normalize(a, eps):
    a += eps
    a_sum = np.sum(a, axis=0)
    a_sum = np.expand_dims(a_sum, axis=0)
    return a / a_sum


def __get_weights(sequence, alphas, nice_illumination, sigma, eps):
    laplacian_kernel = np.array(([0, 1, 0], [1, -4, 1], [0, 1, 0]), dtype=np.float32)

    weights = []

    for img_id in range(len(sequence)):
        img = sequence[img_id].astype(np.float32) / 255

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contrast = cv2.filter2D(gray, -1, laplacian_kernel, borderType=cv2.BORDER_REPLICATE)
        contrast = np.abs(contrast)

        mean = np.mean(img, axis=-1)
        mean = np.expand_dims(mean, axis=-1)
        mean = np.repeat(mean, 3, axis=-1)
        saturation = np.sqrt(np.mean(np.power(img - mean, 2), axis=-1))

        well_exposedness = np.exp(-0.5 / (sigma * sigma) * np.sum(np.power(img - nice_illumination, 2), axis=-1))

        cur_weight = np.power(contrast, alphas[0]) *\
                     np.power(saturation, alphas[1]) *\
                     np.power(well_exposedness, alphas[2])
        weights.append(cur_weight)

    weights = np.stack(weights, axis=0)
    weights = __normalize(weights, eps)

    return weights


def naive_fusion(sequence, alphas=(1.0, 1.0, 1.0), nice_illumination=0.5, sigma=0.2, eps=1e-7):
    weights = __get_weights(sequence, alphas, nice_illumination, sigma, eps)

    weights = np.expand_dims(weights, axis=-1)
    weights = np.repeat(weights, 3, axis=-1)

    result = np.sum(sequence * weights, axis=0)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def gaussian_fusion(sequence, gaussian_ksize=(59, 59), gaussian_sigma=40.0, alphas=(1.0, 1.0, 1.0),
                    nice_illumination=0.5, sigma=0.2, eps=1e-7):
    weights = __get_weights(sequence, alphas, nice_illumination, sigma, eps)

    weights = np.stack([cv2.GaussianBlur(e, gaussian_ksize, gaussian_sigma) for e in weights], axis=0)
    weights = __normalize(weights, eps)

    weights = np.expand_dims(weights, axis=-1)
    weights = np.repeat(weights, 3, axis=-1)

    result = np.sum(sequence * weights, axis=0)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def laplacian_fusion(sequence, layers=8, gaussian_ksize=(5, 5), gaussian_sigma=1.0, alphas=(1.0, 1.0, 1.0),
                     nice_illumination=0.5, sigma=0.2, eps=1e-7):
    weights = __get_weights(sequence, alphas, nice_illumination, sigma, eps)

    def get_gaussian_pyramid(img):
        gaussian_pyramid = [img]

        for layer_id in range(1, layers):
            blurred = cv2.GaussianBlur(gaussian_pyramid[-1], gaussian_ksize, gaussian_sigma)
            blurred = blurred[::2, ::2]
            gaussian_pyramid.append(blurred)

        return gaussian_pyramid

    def get_laplacian_pyramid(img):
        gaussian_pyramid = get_gaussian_pyramid(img)
        gaussian_pyramid = [e.astype(np.int16) for e in gaussian_pyramid]
        laplacian_pyramid = []

        for layer_id in range(0, layers - 1):
            h, w = gaussian_pyramid[layer_id].shape[:2]
            upsampling = cv2.resize(gaussian_pyramid[layer_id + 1], (w, h))
            laplacian_pyramid.append(gaussian_pyramid[layer_id] - upsampling)

        laplacian_pyramid.append(gaussian_pyramid[-1])

        return laplacian_pyramid

    sequence_weights_pyramids = [get_gaussian_pyramid(weights[img_id]) for img_id in range(len(sequence))]
    sequence_laplace_pyramids = [get_laplacian_pyramid(sequence[img_id]) for img_id in range(len(sequence))]

    fused_laplacian_pyramid = [np.sum([sequence_laplace_pyramids[img_id][layer_id] *
                                       np.expand_dims(sequence_weights_pyramids[img_id][layer_id], axis=-1)
                                       for img_id in range(len(sequence))], axis=0) for layer_id in range(layers)]

    result = fused_laplacian_pyramid[-1]

    for layer_id in range(layers - 1, 0, -1):
        h, w = fused_laplacian_pyramid[layer_id - 1].shape[:2]
        result = cv2.resize(result, (w, h))
        result += fused_laplacian_pyramid[layer_id - 1]

    result = np.clip(result, 0, 255).astype(np.uint8)

    return result
