import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
import librosa
import scipy.linalg as la
import os

fs = 48e3 # [kHz]
duration = 10 # [s]
t = np.linspace(0,duration, int(fs*duration), endpoint=False)
n_sync = int(fs/2)
n_delay = int(5*fs)
N = fs * duration

# rectangular signal (for synching)
a = 1
f = 100
u_rec = a*np.sign(np.sin(2*np.pi*f*t))

# loading the exp signal
f_0 = 20
f_1 = 20e3
k = (f_1-f_0)**(1/duration) 
u_exp = np.sin(2*np.pi*f_0*((k**t)-1)/np.log(k))

y_exp, f_exp = librosa.load(r"exp.wav", sr=48000)

# adding the Synchronization:
u_sync = np.concatenate([u_exp[:n_sync], np.zeros(n_delay)])
t_wsync = np.linspace(0, int(len(u_sync/fs)), len(u_sync) , endpoint=False)

u_exp_sync = np.concatenate([u_sync, u_exp])
t_sync = np.concatenate([t_wsync, t])

# plt.figure()
# plt.plot(u_exp_sync)
# plt.plot(y_exp)

# x = u_exp_sync[(n_sync + n_delay):]
# y = y_exp[(n_sync + n_delay):(n_sync + n_delay + int(fs*duration))]

# plt.figure()
# plt.plot(x)
# plt.plot(y)
# print(x.shape, y.shape)

'''
Synchronization:
take the first second and calculate the crosscorrelation to recieve the lag between the 2 signals
take the lag in account for the real excitations
we have to consider two lists of input signals (1. without synch, 2. with synch)
implement for loop for 4
'''

def sync(u, y, n_sync, n_delay, n_buffer=0):
    # can lead to issues if lag is negative, just let the recording run long enough
    # u = u_synch_test
    # u_synch = u[:int(n_synch+n_puffer)]
    u_sync = u_rec[:n_sync]
    # y = y_synch_test 
    y_sync = y[:int(n_sync+n_buffer)] 
    corr = sp.correlate(u_sync, y_sync, mode='full')
    lags = sp.correlation_lags(len(u_sync), len(y_sync), mode='full')
    lag = lags[np.argmax(corr)] # lag positive -> y has lag to x -> y move to left for lag
    print(lag)
    u_real = u[int(n_sync+n_delay):int(n_sync+n_delay+duration*fs)]
    y_real = y[int(n_sync+n_delay-lag):int(n_sync+n_delay+duration*fs-lag)]
    # u_real = u
    # y_real = y[int(-lag):int(-lag)]
    return u_real, y_real

x, y = sync(u_exp_sync, y_exp, n_sync, n_delay)

plt.figure()
plt.plot(x)
plt.plot(y)
print(x.shape, y.shape)

'''
q3 finding optimal M
'''
sigma_y2 = np.mean(np.abs(y)**2)  # E{|y|^2}

X = np.fft.fft(x)  # FFT of signal
Y = np.fft.fft(y)

rxx = np.fft.ifft(X * np.conj(X)) / N # autocorrelation SEQUENCE
pxy = np.fft.ifft(Y * np.conj(X)) / N

# Jo_tab = []
# M_tab = [1, 100, 200, 500, 1000, 2000, 5000, 8000, 10000]

# for M in M_tab:
    
#     R = la.toeplitz(rxx[:M])  # autocorrelation matrix 
#     p = np.conj(pxy)[:M]  # cross correlation vector
    
#     Jo = sigma_y2 - np.conj(p) @ la.solve(R, p)  # sigma_y^2 - p^H R^-1 p
#     Jo_tab.append(np.real(Jo))  # imaginary parts are e-14 and are just floating point error basically
    
# plt.figure()
# plt.plot(M_tab, Jo_tab)
# plt.xlabel("M")
# plt.ylabel("Jo")
# print(Jo_tab)

'''
simulating q4
'''

M = 10000  # optimal M from previous question

R = la.toeplitz(rxx[:M])  # autocorrelation matrix 
p = np.conj(pxy)[:M]  # cross correlation vector

# characteristic equation: Rw_o = p
w_o = la.solve(R, p)
# print(w_o)  # this is the h(n) 

'''
H_hat tells us how the room modifies each frequency, so basically which 
frequencies are amplified, attenuated, or delayed etc 
'''
H_hat = np.fft.fft(w_o)  # the frequency response
# plt.figure()
# plt.plot(H_hat)

'''
E = 1 - sigma_yhato^2 / sigma_y^2
sigma_y^2 = E{|y|^2} --> we're estimating this with just a mean
sigma_yhato^2 = E{|y_hato|^2} = p^H w_o
'''
sigma_y_hato2 = np.conj(p) @ w_o
MSE_normalizado = 1 - sigma_y_hato2 / sigma_y2
print(MSE_normalizado.real)

# '''
# compare between the different signals we recorded --> conclusions about which
# ones excite the frequency spectrum more hence allow us to build a better filter
# for the room
# '''

# '''
# q5. just discuss which had lowest normalised MSE vs. what we expected
# q6. best h(n) would be one with lowest MSE most likely --> the music recorded
# in the room can be described as y(n) = x(n) * h(n) where x(n) is the og music
# and its convoluted by the room so we want an inverse filter
# since convolution in the time domain is multiplication in the frequency domain
# so Y(f) = X(f) H(f) --> X(f) = Y(f) / H(f)
# '''
# Y = np.fft.fft(y)  # putting signal into frequency domain
# X_hat = Y / H_hat  # "unconvoluting" the signal in the frequency domain
# x_hat = np.fft.ifft(X_hat)  # bringing it back to the time domain

