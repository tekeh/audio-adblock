import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from librosa import power_to_db, load
from librosa import load
from librosa.feature import mfcc, melspectrogram
import librosa.display
from sklearn.decomposition import PCA
from sklearn.cluster import k_means
import subprocess

#
fname   = './ftnb_2021-06-15.wav'
y, sr   = load(fname)
time    = np.arange(0, len(y))/sr

hop_secs    = 2
win_secs    = 2
hop_length  = int(hop_secs*sr) ## 1 s * sr samples s^-1
win_length  = int(win_secs*sr)
mfccs   = mfcc(y,               sr=sr, hop_length=hop_length, n_fft=win_length)
mels    = melspectrogram(y=y,   sr=sr, hop_length=hop_length, n_fft=win_length)

##### Plot the melspec along with the classifications for visualisation
fig, (ax0, ax1, ax2) = plt.subplots(3,1, sharex=True)
fig.canvas.mpl_connect('close_event', lambda event: play_pod_proc.kill())
title = ax0.text(0.5, 20.2, "")
## MFCCS and Mel Spectrogram 
#librosa.display.specshow(mfccs, sr=sr, x_coords = time[::hop_length], ax=ax0)
#librosa.display.specshow(lr.power_to_db(mels,ref=np.max), x_coords=time[::hop_length], y_axis='mel', fmax=4096, ax=ax1)
## Clustering  
n_clusters = 6 ##3 or 4 seems decent
ax1.set_xlim([0, max(time[::hop_length])])
km = k_means(mfccs.T, n_clusters)
#ax2.plot(time[::hop_length], km[1])
#ax2.set_xlim([0, max(time[::hop_length])])
#ax0.set_colorbar(format='%+2.0f dB')


def init():
    line0, = librosa.display.specshow(mfccs, sr=sr, x_coords = time[::hop_length], ax=ax0)
    line1, = librosa.display.specshow(power_to_db(mels,ref=np.max), x_coords=time[::hop_length], y_axis='mel', fmax=4096, ax=ax1)
    line2, = ax2.plot(time[::hop_length], km[1])
    return [line0, line1, line2]

def animate(i):
    line0 = librosa.display.specshow(mfccs, sr=sr, x_coords = time[::hop_length], ax=ax0)
    line1 = librosa.display.specshow(power_to_db(mels,ref=np.max), x_coords=time[::hop_length], y_axis='mel', fmax=4096, ax=ax1)
    
    ax2.cla()
    line2, = ax2.plot(time[::hop_length], km[1])
    line3, = ax2.plot([i+1,i+1], [0,max(km[1])], 'g')
    title.set_text(f"{i} secs")
    #ax2.figure.canvas.draw()
    if i==1: ## ==0 seems to have wired overlap issue -- investigate
        global play_pod_proc
        play_pod_proc = subprocess.Popen(['cvlc', fname])

    return [line2, line3, title]

ani = FuncAnimation(fig, animate, interval=1000, blit=True) ## 1 second intervals
plt.show()

## Try a naive PCA
n_components = 3
pca = PCA(n_components=n_components)
pca.fit(mfccs.T)
pcomps = pca.components_

## Do clustering in full dimension 
# Defined above n_clusters = 4
cols = [[1,0,0], [0,1,0], [0,0,1], [0,0,0], [1,1,0],[1,0,1], [0,1,1], [0.5,0.5,0.5]]
kout = k_means(mfccs.T, n_clusters)
categories  = kout[1]
colors      = np.array([cols[cat] for cat in categories])

## plot the scatter to see if there are nicely separated clusters in 3D
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

x1 = (mfccs.T).dot(pcomps[0])
x2 = (mfccs.T).dot(pcomps[1])
x3 = (mfccs.T).dot(pcomps[2])
ax.scatter(x1,x2,x3, c=colors)
plt.show()
