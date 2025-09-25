import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
import librosa
import scipy.linalg as la
import os

# basics
fs = 48e3 # [kHz]
duration = 10 # [s]
t = np.linspace(0,duration, int(fs*duration), endpoint=False)
n_sync = int(fs/2)
n_delay = int(5*fs)
N = fs * duration

# storing the signals
u_dict = {}
y_dict = {}
keys = []

# exponential signal (exp)
f_0 = 20
f_1 = 20e3
k = (f_1-f_0)**(1/duration) 
u_exp = np.sin(2*np.pi*f_0*((k**t)-1)/np.log(k))
y_exp, f_exp = librosa.load(r"exp.wav", sr=48000)
u_dict["exp"] = u_exp
y_dict["exp"] = y_exp
keys.append("exp")

# linear signal (lin)
f_0 = 20
f_1 = 20e3
k = (f_1-f_0)/duration 
u_lin = np.sin(2*np.pi*f_0 + np.pi*k*t**2)
y_lin, f_lin = librosa.load(r"lin.wav", sr=48000)
u_dict["lin"] = u_lin
y_dict["lin"] = y_lin
keys.append("lin")

# music signal (music)
u_music, f_music = librosa.load(r"music_seg.wav", sr=48000)
y_music, f_music = librosa.load(r"music.wav", sr=48000)
u_dict["music"] = u_music
y_dict["music"] = y_music
keys.append("music")

# rectangular signal (rec)
a = 1
f = 100
u_rec = a*np.sign(np.sin(2*np.pi*f*t))
y_rec, f_rec = librosa.load(r"rec.wav", sr=48000)
u_dict["rec"] = u_rec
y_dict["rec"] = y_rec
keys.append("rec")

# voice signal (voice)
u_voice, f_voice = librosa.load(r"voice_seg.wav", sr=48000)
y_voice, f_voice = librosa.load(r"voice.wav", sr=48000)
u_dict["voice"] = u_voice
y_dict["voice"] = y_voice
keys.append("voice")

# white noise signal (wn)
u_wn = np.random.normal(0,1, len(t))
y_wn, f_wn = librosa.load(r"wn.wav", sr=48000)
u_dict["wn"] = u_wn
y_dict["wn"] = y_wn
keys.append("wn")

# adding the synching section
u_sync = np.concatenate([u_exp[:n_sync], np.zeros(n_delay)])
t_wsync = np.linspace(0, int(len(u_sync/fs)), len(u_sync) , endpoint=False)
t_sync = np.concatenate([t_wsync, t])

def sync(u, y, n_sync, n_delay, n_buffer=0):
    # can lead to issues if lag is negative, just let the recording run long enough
    u_sync = u_rec[:n_sync]
    y_sync = y[:int(n_sync+n_buffer)] 
    corr = sp.correlate(u_sync, y_sync, mode='full')
    lags = sp.correlation_lags(len(u_sync), len(y_sync), mode='full')
    lag = lags[np.argmax(corr)] # lag positive -> y has lag to x -> y move to left for lag
    # print(lag)
    u_real = u[int(n_sync+n_delay):int(n_sync+n_delay+duration*fs)]
    y_real = y[int(n_sync+n_delay-lag):int(n_sync+n_delay+duration*fs-lag)]
    return u_real, y_real

u_real_dict = {}
y_real_dict = {}

for key in keys:
    u_dict[key] = np.concatenate([u_sync, u_dict[key]])  # adding synching section

    # plt.figure()
    # plt.plot(u_dict[key])
    # plt.plot(y_dict[key])
    # plt.title(f"{key} signal with synching")

    u_real_dict[key], y_real_dict[key] = sync(u_dict[key], y_dict[key], n_sync, n_delay)

    # plt.figure()
    # plt.plot(u_real_dict[key])
    # plt.plot(y_real_dict[key])
    # plt.title(f"{key} signal")
    
'''
q3 finding optimal M
'''
# sigma_y2_lin = np.mean(np.abs(y_real_dict["lin"])**2)  # E{|y|^2}

# U_lin = np.fft.fft(u_real_dict["lin"])  # FFT of signal
# Y_lin = np.fft.fft(y_real_dict["lin"])

# ruu_lin = np.fft.ifft(U_lin * np.conj(U_lin)) / N # autocorrelation SEQUENCE
# puy_lin = np.fft.ifft(Y_lin * np.conj(U_lin)) / N

# Jo_tab = []
# M_tab = [1, 100, 200, 500, 1000, 2000, 5000, 8000, 10000]

# for M in M_tab:
    
#     R_lin = la.toeplitz(ruu_lin[:M])  # autocorrelation matrix 
#     p_lin = np.conj(puy_lin)[:M]  # cross correlation vector
    
#     Jo = sigma_y2_lin - np.conj(p_lin) @ la.solve(R_lin, p_lin)  # sigma_y^2 - p^H R^-1 p
#     Jo_tab.append(np.real(Jo))  # imaginary parts are e-14 and are just floating point error basically
    
# plt.figure()
# plt.plot(M_tab, Jo_tab)
# plt.title("M-Jo for linear sweep")
# plt.xlabel("M")
# plt.ylabel("Jo")
# print(Jo_tab)


'''
simulating q4
'''

M = 10000  # optimal M from previous question

MSE_tab = []
wo_tab = []

for key in keys:
    sigma_y2 = np.mean(np.abs(y_real_dict[key])**2)  # E{|y|^2}

    U = np.fft.fft(u_real_dict[key])  # FFT of signal
    Y = np.fft.fft(y_real_dict[key])

    ruu = np.fft.ifft(U * np.conj(U)) / N # autocorrelation SEQUENCE
    puy = np.fft.ifft(Y * np.conj(U)) / N
    
    R = la.toeplitz(ruu[:M])  # autocorrelation matrix 
    p = np.conj(puy)[:M]  # cross correlation vector
    
    # characteristic equation: Rw_o = p
    w_o = la.solve(R, p)
    wo_tab.append(w_o)
    # print(w_o)  # this is the h(n) 
    
    '''
    H_hat tells us how the room modifies each frequency, so basically which 
    frequencies are amplified, attenuated, or delayed etc 
    '''
    H_hat = np.fft.fft(w_o)  # the frequency response
    plt.figure()
    plt.plot(H_hat)
    plt.title(f"H_hat for {key} signal")
    
    '''
    E = 1 - sigma_yhato^2 / sigma_y^2
    sigma_y^2 = E{|y|^2} --> we're estimating this with just a mean
    sigma_yhato^2 = E{|y_hato|^2} = p^H w_o
    '''
    sigma_y_hato2 = np.conj(p) @ w_o
    MSE_normalizado = 1 - sigma_y_hato2 / sigma_y2
    MSE_tab.append(MSE_normalizado.real)
    print(MSE_normalizado.real)
    
print(MSE_tab)
