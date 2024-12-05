import yfinance as yf
import numpy as np
import pywt as pywt
from PyEMD import CEEMDAN
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller


class StockData():
    def __init__(self, ticker, start_dt, end_dt):
        self.ticker = ticker
        self.start_dt = start_dt
        self.end_dt = end_dt
        self._clean_ret = None  
        self._noisy_ret = None  

    def calculate_returns(self, data):
        returns = data.pct_change()
        returns = returns.dropna()
        return returns

    def returns(self, add_noise=False, noise_mean=0, noise_std_dev=1):
        if (self._clean_ret is None and add_noise==False) or (self._noisy_ret is None and add_noise==True):
            data = yf.download(self.ticker, start=self.start_dt, end=self.end_dt)["Adj Close"]
            if add_noise==False:
                    self._clean_ret = self.calculate_returns(data)
            else:
                num_samples = len(data)
                gaussian_noise = np.random.normal(noise_mean, noise_std_dev, num_samples)
                noisy_data = data + gaussian_noise
                self._noisy_ret = self.calculate_returns(noisy_data)
                
        if add_noise==False:
            return self._clean_ret
        else:
            return self._noisy_ret

    def get_snr(self, clean_signal, noisy_signal):
        clean_signal_power = np.mean(clean_signal**2)
        noise = clean_signal - noisy_signal
        noisy_signal_power = np.mean(noise**2)
        return clean_signal_power / noisy_signal_power


class ICEEMDAN_denoise():
    def __init__(self, clean_data, noisy_data):
        self.clean_ret = clean_data
        self.noisy_ret = noisy_data

        self._get_IMFs = None
        self._get_separation_point = None\

    # calculate IMFs:
    def get_imfs(self):
        if self._get_IMFs is None:
            s = self.noisy_ret.values
            t = self.noisy_ret.index

            # EEMD on s
            ceemdan = CEEMDAN()
            c_IMF = ceemdan(s)
            nCIMF = c_IMF.shape[0]
            IMFs = c_IMF
            sum_IMFs = np.sum(IMFs, axis=0)
            R = s - sum_IMFs 
            self._get_IMFs = IMFs, R
            return self._get_IMFs
        else:
            return self._get_IMFs

    # test for zero mean:
    def population_mean_test(self, data, alpha=0.01):
        x_bar = np.mean(data)
        s = np.std(data)
        n = len(data)
        t_statistic = x_bar / (s / np.sqrt(n))

        # critical values   
        t_critical_lower = stats.t.ppf(alpha/2, n-1)
        t_critical_upper = stats.t.ppf(1-alpha/2, n-1)
        
        return (t_statistic >= t_critical_lower and t_statistic <= t_critical_upper)
    
    # test for stationarity (ADF):
    def adf_test(self, data, alpha=0.01):
        result = adfuller(data)
        p_value = result[1]
        return p_value < alpha

    # find the separation point:
    def get_separation_point(self):
        if self._get_separation_point is None:
            separation_point = None
            test_results = []
            IMFs, R = self.get_imfs()

            for i in range(1, len(IMFs)):
                # Condition 1: Population mean test for each IMF component
                condition_1 = all(self.population_mean_test(imf) for imf in IMFs[:i])
                
                # Condition 2: Population mean test for the sum of IMF components
                condition_2 = self.population_mean_test(np.sum(IMFs[:i], axis=0))
                
                # Condition 3: ADF test for each IMF component
                condition_3 = all(self.adf_test(imf) for imf in IMFs[:i])
                
                # Condition 4: ADF test for the sum of IMF components
                condition_4 = self.adf_test(np.sum(IMFs[:i], axis=0))
                
                test_results.append((i, condition_1, condition_2, condition_3, condition_4))

            # we not find the separation point that satisfies all conditions
            for i, condition_1, condition_2, condition_3, condition_4 in test_results:
                if condition_1 and condition_2 and condition_3 and condition_4:
                    continue
                else:
                    separation_point = i-1
                    break
            self._get_separation_point = separation_point
        return self._get_separation_point

    # separate into x_noise and x_non_noise:
    def get_x_noise(self):
        separation_point = self.get_separation_point()
        IMFs, _ = self.get_imfs()
        return np.sum(IMFs[:separation_point+1], axis=0)

        # IMFs, R = self.get_imfs()
        # return np.sum(IMFs[separation_point:], axis=0) + R



    def get_x_non_noise(self):
        separation_point = self.get_separation_point()
        IMFs, R = self.get_imfs()
        return np.sum(IMFs[separation_point:], axis=0) + R

        # IMFs, _ = self.get_imfs()
        # return np.sum(IMFs[:separation_point+1], axis=0)
        

    # different threshold values:
    def rigrsure(self, coeffs, noise_std_dev):
        risks = []
        
        for coeff in coeffs:
            coeff_sq = np.abs(coeff) ** 2
            
            N = coeff.size
        
            sq_norm = np.sum(coeff_sq)
            
            threshold = np.sqrt(sq_norm) * np.sqrt(2 * np.log(N))
            
            risk = N * noise_std_dev ** 2 - sq_norm + 2 * N * noise_std_dev ** 2
            
            risks.append(risk)
        
        return risks

        return thresholded_coeffs

    def heursure(self, coeffs, noise_std_dev):
        thresholds = []
        
        for coeff in coeffs:
            threshold = np.sqrt(2 * np.log(len(coeff))) * noise_std_dev
            thresholds.append(threshold)
        
        return thresholds
    
    def minimax(self, coeffs, noise_std_dev):
        thresholds = []
        
        for coeff in coeffs:
            threshold = np.sqrt(2 * np.log(len(coeff))) * noise_std_dev / 0.6745
            thresholds.append(threshold)
        
        return thresholds

    def sqtwolog(self, coeffs, noise_std_dev):
        thresholds = []
        
        for coeff in coeffs:
            threshold = np.sqrt(2 * np.log(len(coeff))) * noise_std_dev
            thresholds.append(threshold)
        
        return thresholds

    def wavelet_denoising(self, wavelet='db4', decomp_level=4, threshold_mode='soft', threshold_value='other'):
        # 1. decomposition into IMFs:
        x_noise = self.get_x_noise()
        coeffs = pywt.wavedec(x_noise, wavelet, level=decomp_level)  

        # 2. thresholding
        if threshold_value=="sqtwolog":
            thresholdLambda = self.sqtwolog(coeffs, np.std(x_noise))
        elif threshold_value=="rigrsure":
            thresholdLambda = self.rigrsure(coeffs, np.std(x_noise))
        elif threshold_value=="heursure":
            thresholdLambda = self.heursure(coeffs, np.std(x_noise))
        elif threshold_value=="minimax":
            thresholdLambda = self.minimax(coeffs, np.std(x_noise))
        else:
            thresholdLambda = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(x_noise)))
            

        # calculate threshold coefficients
        if threshold_mode in ['soft', 'hard', 'garrote', 'greater', 'less']:
            if threshold_value=="other":
                threshold = [pywt.threshold(c, thresholdLambda, mode=threshold_mode) for c in coeffs]
            else:
                threshold = [pywt.threshold(c, t, mode=threshold_mode) for c, t in zip(coeffs, thresholdLambda)]
        else:
            if threshold_value=="other":
                threshold = [threshold_mode(wavelet_coefficients=c, L=thresholdLambda) for c in coeffs]
            else:
                threshold = [threshold_mode(wavelet_coefficients=c, L=t) for c,t in zip(coeffs, thresholdLambda)]

        x_vec = pywt.waverec(threshold, wavelet)

        epsilon = x_noise - x_vec

        x_tilde = self.get_x_non_noise() + x_vec

        return x_tilde

    def get_snr(self, cleaned_signal):
        clean_signal = self.clean_ret
        noisy_signal = self.noisy_ret
        clean_signal_power = np.sqrt(np.sum(np.square(clean_signal)))

        noisy_noise = clean_signal - noisy_signal
        noisy_signal_power = np.sqrt(np.sum(np.square(noisy_noise)))

        cleaned_noise = clean_signal - cleaned_signal
        cleaned_signal_power = np.sqrt(np.sum(np.square(cleaned_noise)))

        return clean_signal_power/noisy_signal_power, clean_signal_power/cleaned_signal_power
