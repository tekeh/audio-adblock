import numpy as np
import matplotlib.pyplot as plt
import librosa as lr
from librosa import load
from librosa.feature import mfcc, melspectrogram
import librosa.display
from sklearn.decomposition import PCA
from sklearn.cluster import k_means
#
y, sr   = lr.load('./ftnb_2021-06-15.wav')
time    = np.arange(0, len(y))/sr

hop_length = int(5*sr) ## 5 s * sr samples s^-1
mfccs   = mfcc(y, sr=sr, hop_length=hop_length)
mels    = melspectrogram(y=y, sr=sr, hop_length=hop_length)

librosa.display.specshow(lr.power_to_db(mels,ref=np.max),y_axis='mel', fmax=2048,x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.show()

librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.show()


## Try a naive PCA
n_components = 3
pca = PCA(n_components=n_components)
pca.fit(mfccs.T)
pcomps = pca.components_

## plot the scatter to see if there are nicely separated clusters in 3D
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

x1 = (mfccs.T).dot(pcomps[0])
x2 = (mfccs.T).dot(pcomps[1])
x3 = (mfccs.T).dot(pcomps[2])
ax.scatter(x1,x2,x3)
plt.show()

## Do clustering in full dimension 
n_clusters = 3
k_means(mfccs.T, n_clusters)
