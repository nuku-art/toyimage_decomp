import pywt
import numpy as np

class WaveletTranslation():
    def __init__(self):
        pass
    
    def sigmoid(x):
        a = 5.0
        b = 0.3
        return 1.0/(1.0 + np.exp(a*(b-x)))
    
    def zero_coeffs_like(co):
        if isinstance(co, np.ndarray):
            return np.zeros_like(co)  # ndarrayの場合
        elif isinstance(co, list):
            return [zero_coeffs_like(c) for c in co]  # リストの場合
        elif isinstance(co, tuple):
            return tuple(zero_coeffs_like(c) for c in co)  # タプルの場合
        else:
            raise TypeError("Unsupported type in coeffs structure")
    
    def translate(self, image, MOTHER_WAVELET='db1', LEVEL=1):
        if isinstance(image, np.ndarray):
            if image.dtype != np.float64:
                image = image.astype(np.float64)
        else:
            image = np.array(image, dtype=np.float64)
        
        coeffs = pywt.wavedec2(image, MOTHER_WAVELET, level=LEVEL)
        zero_coeffs = self.zero_coeffs_like(coeffs)
        cc_list = ['cH', 'cV', 'cD']

        total_reconstruct = np.zeros_like(image)
        image_sum = np.zeros_like(image)
        check_array = np.zeros_like(image[0:4,0:4])
        edge_sum = np.zeros_like(image)
        