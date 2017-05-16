import sys
import scipy
import scipy.io.wavfile
from scipy import signal, cluster, stats
import sklearn.cluster
from hmmlearn import hmm
import time
import string
import util

import numpy as np
from numpy.fft import rfft, hfft

"""
Decode acoustic ciphertexts.

The core steps of decoding are as follows:

    (1) Isolate the start of each chirp.
    (2) Extract features for each chirp.
    (3) Cluster/label the set of chirp feature vectors.
    (4) Use Hidden Markov Model to predict most likely label for each cluster [Optional]
"""

def detect_chirp_starts(samples, sample_rate):
    """ Given an audio signal as an array 'samples', detect
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

    # We extract sound peaks by looking at signal energy. The 'threshold' value can just be
    # determined empirically with a little inspection/trial & error if peaks in the original
    # signal are well defined and the signal is not too noisy.
    threshold = 0.1
    chirp_start_positions = []

    # Used to hold off detection after a peak.
    holdoff_until_sample = 0

    # Extract chirp starts. Hold off after fidning a peak so we don't double count.
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
    """ Given an audio signal, 'samples', and the positions of 'chirps' within the 'samples' array,
        compute FFT features by looking at a window around each chirp. """

    # Compute the FFT features for each chirp.
    chirp_feature_ffts = compute_fft_windows(samples, sample_rate, chirp_start_positions)
    chirp_feature_ffts/np.max(chirp_feature_ffts,axis=1).reshape(-1,1)

    # Compute an array of the most dominant frequencies to use as a feature vector.
    # The number of frequencies to extract can be chosen based on how many interesting
    # frequencies you expect to be in the signal :)
    chirp_feature_vectors = []
    num_dom_freqs = 10
    for row in chirp_feature_ffts:
        row = row/np.max(row)
        peak_vec = np.zeros(num_dom_freqs)
        peak_positions = []
        # This threshold can be determined empirically.
        threshold = 0.78
        i=0
        while i < len(row):
            if row[i]>threshold and len(peak_positions)<num_dom_freqs:
                peak_positions.append(i)
                i+=5 # jump forward a bit so we don't count a peak twice.
            else:
                i+=1

        peak_vec[:len(peak_positions)] = peak_positions
        chirp_feature_vectors.append(peak_vec)

    return np.array(chirp_feature_vectors), None

def train_hmm(labels):
    """ Given a sequence of labels, predict the msot common mapping of labels to english letters."""
    init_probs = np.load("english_letter_probs.npy")
    trans_probs = np.load("english_letter_trans_probs.npy")

    # Number of alphabet letters plus a SPACE.
    num_states=27
    model = hmm.MultinomialHMM(n_components=num_states,
    							verbose=False,
    							params='e',
    							init_params='e',
    							tol=0.0001,
    							n_iter=350)

    model.startprob_ = np.array(init_probs)
    model.transmat_ = np.array(trans_probs)
    model.emissionprob_ = np.random.rand(num_states, num_states)

    labels = labels.reshape(-1,1)

    model.emissionprob_ = np.random.rand(num_states, num_states)
    model.fit(labels)
    preds = model.predict(labels)
    score = model.score(labels)

    return score, preds

def hmm_predict(labels, num_iters=8):
    scores, preds = [], []

    # EM algorithm performance is sensitive to initial conditions, so we try several times
    # and pick the best model.
    for n in range(num_iters):
        score, pred = train_hmm(labels)
        scores.append(score)
        preds.append(pred)
        print("".join([util.pos_to_letter(p) for p in pred]))

    best_ind = scores.index(max(scores))
    best_pred = preds[best_ind]
    best_pred_str = "".join([util.pos_to_letter(p) for p in best_pred])
    return best_pred_str

if __name__ == '__main__':
    # Seed randomness.
    np.random.seed(int(time.time()))

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
    chirp_labels = skmeans.labels_

    pred_str = "".join([util.pos_to_letter(l) for l in chirp_labels]).lower()
    print(pred_str)

    # (4) Hidden Markov Model Label Inference. [Optional]
    pred = hmm_predict(chirp_labels)


