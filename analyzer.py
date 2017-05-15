import sys
import scipy
import scipy.io.wavfile
from scipy import signal, cluster, stats
from scipy.cluster.vq import vq, kmeans, whiten
import sklearn.cluster
import time
import string
import util

import numpy as np
from numpy.fft import rfft, hfft

import matplotlib.pyplot as plt


""" Acoustic Emanations

	Script to decode acoustic ciphertexts.

	(1) Isolate the start of each chirp.
	(2) Extract features for each chirp.
	(3) Cluster/label the set of chirp feature vectors.
	(4) Use Hidden Markov Model to predict most likely label for each cluster

"""

def detect_chirp_starts(samples, sample_rate):
    """ Given an audio signal, given as an array 'samples', detect
        the start of chirps in the signal and return an array of their
        positions in the given 'samples' array. """

    # Compute spectrogram of signal
    window_ms = 5
    samples_per_segment = int(sample_rate * (window_ms / 1000))
    overlap = int(samples_per_segment / 8)
    specgram_opts = {
        "nperseg": samples_per_segment,
        "noverlap": overlap,  # in samples
    }

    sample_freqs, segment_times, specgram = signal.spectrogram(samples, **specgram_opts)
    energy_signal = np.sum(specgram, axis=0)

    def window_to_sample_pos(i):
        """ Convert a window index to an index in the original 'samples' array """
        return i * (specgram_opts["nperseg"] - specgram_opts["noverlap"])

    # Extract sound peaks by looking at signal energy. The threshold value can just be
    # determined empirically if peaks are well defined and the signal is not very noisy.
    threshold = 0.1
    chirp_start_positions = []

    # Used to hold off detection after a peak.
    holdoff_until_sample = 0

    # Extract chirp starts. If two peaks are less than 'chirp_holdoff_ms' apart,
    # assume they are a push/release pair and only keep the first one i.e. the
    # 'push'
    i = 0
    chirp_holdoff_ms = 150

    while i < len(energy_signal):
        if window_to_sample_pos(i) < holdoff_until_sample:
            i += 1
        else:
            diff = energy_signal[i] - energy_signal[i - 1]
            if energy_signal[i] > threshold:
                sample_pos = window_to_sample_pos(i)
                chirp_start_positions.append(sample_pos)
                holdoff_until_sample = sample_pos + \
                    util.millis_to_samples(sample_rate, chirp_holdoff_ms)
            i += 1

    return np.array(chirp_start_positions), energy_signal

def compute_fft_windows(samples, sample_rate, start_positions):
    # Length of a window in samples
    feature_windows = []
    feature_window_ms = 10  # milliseconds
    shift_ms = 0
    shift_samples = util.millis_to_samples(sample_rate, shift_ms)
    feature_window_length = util.millis_to_samples(
        sample_rate, feature_window_ms)

    for start_pos in start_positions:
        st, end = (start_pos - shift_samples,
                   int(start_pos + feature_window_length))
        feature_windows.append(samples[st:end])

    # Use Fourier coefficients of each window as our feature vectors.
    feature_windows = np.array(feature_windows)
    feature_ffts = np.absolute(rfft(feature_windows, norm='ortho', axis=1))

    return feature_ffts

def make_chirp_fvectors(samples, sample_rate, chirp_start_positions):
    chirp_feature_ffts = compute_fft_windows(samples, sample_rate, chirp_start_positions)
    chirp_feature_ffts/np.max(chirp_feature_ffts,axis=1).reshape(-1,1)

    # Extract array of most dominant frequencies to use as a feature vector.
    chirp_feature_vectors = []
    num_dom_freqs = 10
    for row in chirp_feature_ffts:
        row = row/np.max(row)
        peak_vec = np.zeros(num_dom_freqs)
        peak_positions = []
        threshold = 0.78
        i=0
        while i < len(row):
            if row[i]>threshold and len(peak_positions)<num_dom_freqs:
                peak_positions.append(i)
                i+=5
            else:
                i+=1

        peak_vec[:len(peak_positions)] = peak_positions
        chirp_feature_vectors.append(peak_vec)

    chirp_feature_vectors = np.array(chirp_feature_vectors)

    return chirp_feature_vectors, None

def train_hmm(labels):
	np.random.seed(int(time.time()))

	init_probs = np.load("english_letter_probs.npy")
	trans_probs = np.load("english_letter_trans_probs.npy")

	num_states=27
	model = hmm.MultinomialHMM(n_components=num_states,
								verbose=False,
								params='e',
								init_params='e',
								tol=0.0001,
								n_iter=2500)

	model.startprob_ = np.array(init_probs)
	model.transmat_ = np.array(trans_probs)
	model.emissionprob_ = np.random.rand(num_states, num_classes)

	# EM algorithm performance is sensitive to initial conditions, so we try several times
	# and pick the best model.
	num_tries = 16
	max_score = None
	max_score_pred = ""

	labels = labels.reshape(-1,1)

	for n in range(num_tries):
		model.emissionprob_ = np.random.rand(num_states, num_classes)
		model.emissionprob_[26,:] = spaces_row
		model.fit(labels)
		preds = model.predict(labels)
		score = model.score(labels)

		pred_str = ("".join([chr(p+65) for p in preds])).replace("["," ")
		print(pred_str)
		print("Score", score)

		# Save the best model
		if max_score==None or score > max_score:
			max_score = score
			max_score_pred = pred_str

		print(model.emissionprob_[26])

	print("Best model:")
	print(max_score_pred)

if __name__ == '__main__':
	# Read in the WAV file.
	wavfile = sys.argv[1]
	sample_rate, samples = scipy.io.wavfile.read(wavfile)

	# (1) Detect the start of each chirp.
	chirp_start_positions, energy_signal = detect_chirp_starts(samples, sample_rate)

	# (2) Extract FFT Features for each chirp.
	chirp_fvectors, _ = make_chirp_fvectors(samples, sample_rate, chirp_start_positions)

	# (3) Cluster/label the set of feature vectors.
	num_classes = 27
	skmeans = sklearn.cluster.KMeans(n_clusters=num_classes).fit(chirp_fvectors)
	key_cluster_labels = skmeans.labels_

	outstr = "".join([chr(l+65) for l in key_cluster_labels]).replace("["," ")
	print(outstr.lower())



# (4) Hidden Markov Model Label Inference. (Optional)


#
# Plot the signal and energy levels as visual aids
#
# import sys
# if len(sys.argv)>2 and sys.argv[2]=="plot":
# 	plt.figure(0)
# 	plt.subplot(211)
# 	plt.plot(t, energy_signal)


# 	chirp_start_pos = [t[ind] for ind in chirp_start_segment_inds]
# 	plt.plot(chirp_start_pos, np.repeat(0,len(chirp_start_segment_inds)), marker='.', markersize=8)
# 	plt.ylabel('Energy Level')

# 	# plt.subplot(212)
# 	# plt.plot(time_pts, samples)
# 	# plt.ylabel('Signal')
# 	plt.show()


