import numpy as np

# threshold a1
def improved_threshold_a1(wavelet_coefficients, L):
    quantized_coefficients = np.zeros_like(wavelet_coefficients)
    for idx, coeff in np.ndenumerate(wavelet_coefficients):
        if abs(coeff) >= L:
            quantized_coefficients[idx] = np.sign(coeff) * ((abs(coeff)**2 - L**2)**0.5)
        else:
            quantized_coefficients[idx] = 0
    return quantized_coefficients

# threshold a2
def improved_threshold_a2(wavelet_coefficients, L):
    quantized_coefficients = np.zeros_like(wavelet_coefficients)
    for idx, coeff in np.ndenumerate(wavelet_coefficients):
        if abs(coeff) >= L:
            quantized_coefficients[idx] = np.sign(coeff) * (abs(coeff) - 2**(L - abs(coeff)))
        else:
            quantized_coefficients[idx] = 0
    return quantized_coefficients

# threshold a3
def improved_threshold_a3(wavelet_coefficients, L):
    quantized_coefficients = np.zeros_like(wavelet_coefficients)
    for idx, coeff in np.ndenumerate(wavelet_coefficients):
        if abs(coeff) >= L:
            quantized_coefficients[idx] = np.sign(coeff) * (abs(coeff) - (2*L) / (np.exp((abs(coeff) - L) / (L)) + 1))
        else:
            quantized_coefficients[idx] = 0
    return quantized_coefficients
