from utils.VRVM import VRVM_regression
##########################
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma, logsumexp, gamma
from numpy import linalg as la
import random
from scipy.stats import random_correlation
import scipy.linalg

####### Example sinc function ##################
noise= 0.2*(-.5+1*np.random.rand((25)))
np.std(noise)

rand_data=np.random.rand((25))
data_raw=np.sort(-10+20*rand_data)
y_data_raw=np.sinc(np.pi**(-1)*data_raw)
y_noisy=y_data_raw+noise


# build the model
N=len(data_raw)
signal_process=VRVM_regression(N=N, input_samples=data_raw, signal=y_noisy,actual_signal=y_data_raw,max_iter=10000,sigma_phi=1)

# fit the signal
signal_process.fit_predict()

# display the result, build the needed plot items
signal_process.display()

# plot the full result
xs=np.linspace(-10,10,300)
plt.figure(figsize=(8,8))
plt.plot(signal_process.xs,signal_process.x_w @ signal_process.kerphi, '--' ,color='red',label='VRVM')
plt.plot(xs,np.sinc(np.pi**(-1)*xs),color='k',label='Actual function Sinc(x)')
plt.plot(data_raw,y_noisy, 'o' ,color='b',label='Noisy data')
plt.legend()
#plt.savefig('test_case_sinc_with_fit.pdf')
plt.show()

# stem plot of the support vectors
w_est=signal_process.weight_estimation
plt.figure(figsize=(6,6))
plt.stem(np.arange(len(w_est)),w_est, linefmt='k--',markerfmt='o')
plt.axhline(0, color='black', linewidth=1)
plt.xlabel("weight number")
plt.ylabel('<w>')
#plt.savefig('stem.pdf')
plt.show()


print(f"Expected value of the noise in the signal {signal_process.noise_estimation:.5f}")

print(f"Actual Noise {np.std(noise)**2 :.5f}")

print(f"MSE {signal_process.MSE:.5f}")

# Display the ELBO
signal_process.display_elbo()