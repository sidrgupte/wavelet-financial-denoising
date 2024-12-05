# Wavelet-Based Signal Denoising for Financial Time Series

This project implements wavelet-based denoising techniques to enhance financial time-series analysis, specifically for S&P 500 returns. The objective is to apply wavelet transforms to reduce noise and improve signal quality.

## Overview
The project decomposes financial time series using wavelet transforms, applies thresholding to remove noise, and reconstructs a cleaner signal. This approach showed significant improvement in the Signal-to-Noise Ratio (SNR) of the financial data.

## Project Structure
- `notebooks/`: Contains the Jupyter notebooks with the wavelet and Fourier transform analysis.
- `code/`: Contains all the functions in structured Python classes
 **Presentation**: A detailed presentation on the methods and results of the wavelet-based signal denoising project can be found [here](presentation/wavelets.pdf).


## Results
- Improved SNR from **0.425** (noisy) to **0.996** (denoised).

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/wavelet-financial-denoising.git
