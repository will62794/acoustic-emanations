from scipy import signal
import scipy.io.wavfile
import sounddevice as sd
import numpy as np
from numpy.fft import rfft, hfft
import matplotlib.pyplot as plt
import time
import sys

"""
Generate acoustic ciphertexts from text files.

An acoustic ciphertext is an audio signal that represents some piece of written
English text. It consists a sequence of "chirps", separated by periods of
silence. Each chirp has a unique signature, which corresponds to a single letter
of the English alphabet, or to a SPACE. To generate chirps, each letter is
assigned a unique 7 bit code. There are 7 fundamental frequencies which compose
every chirp. The presence of a frequency in a letter's chirp is determined by
whether its index is 'on', i.e. a 1 bit in the letter's code.

"""

SAMPLE_RATE = 44100


def int_to_bitvector(num, bits):
	return [int(b) for b in bin(num)[2:].zfill(bits)]

def alphabet_ind_to_char(i):
	if i==26:
		return " "
	return chr(i+97)

# The set of fundamental frequencies for ciphertext beeps.
# Each letter is assigned a beep which is some linear combination
# of these frequencies.
fund_freqs = [800, 1600, 2400, 3000, 3800, 4600, 5400]
offset = 15
num_bits = 7
letter_freqs = {alphabet_ind_to_char(n):int_to_bitvector((n+offset),num_bits) for n in range(0,27)}


def gen_beep(sample_rate, freq, length_ms, ampl):
	""" Generate a sinusoidal wave beep at a single frequency. """
	length_secs = length_ms/1000
	t = np.arange(0, length_secs, 1/sample_rate)

	return signal.chirp(t, freq, length_secs, freq)*ampl

def gen_composite_beep(sample_rate, length_ms, freqs, freq_ampls, add_noise=True):
	""" Generate a sinusoidal beep composed of multiple frequencies. """
	assert len(freqs)==len(freq_ampls)

	# Add a small amount of noise
	def rand_scale():
		return np.random.rand()*0.2 + 0.7

	# We create signal by summing up sine wave frequencies at specified amplitudes
	all_signals = np.array([gen_beep(sample_rate, freq, length_ms, freq_ampls[i]*rand_scale()) for (i,freq) in enumerate(freqs)]).T
	signal_composite = np.sum(all_signals, axis=1)

	# Add signal noise if specified.
	if add_noise:
		noise = np.random.normal(0, 0.5, len(signal_composite))
		out_signal = signal_composite + noise
	else:
		out_signal = signal_composite

	out_ampl = 0.7
	return out_signal/np.max(out_signal) * out_ampl

def gen_cipher_signal(text, beep_length, pause_length):
	""" Given a string of text, construct a signal composed of sequential beeps interspersed
	with pauses where beeps are short signals composed of a linear combination of the fundamentel frequencies
	for each letter. """

	beeps = []

	for letter in text:
		freq_ampls = letter_freqs[letter]

		# Generate random amplitudes and our letter tone.
		sig = gen_composite_beep(SAMPLE_RATE, beep_length_ms, fund_freqs, freq_ampls, add_noise=False)

		# silence (zero amplitude) with some variability.
		random_pause_length_ms = (np.random.rand()*80) + pause_length_ms
		pause = gen_beep(SAMPLE_RATE, 100, random_pause_length_ms, 0)

		beeps.append(sig)
		beeps.append(pause)

	return np.concatenate(beeps)

if __name__ == '__main__':
	# Read in plain text.
	textfile = sys.argv[1]
	text = open(textfile).read().lower()

	beep_length_ms = 80
	pause_length_ms = 130

	# Generate the acoustic cipher text.
	cipher_signal = gen_cipher_signal(text, beep_length_ms, pause_length_ms)
	noise = np.random.normal(0, 0.05, len(cipher_signal))

	# Convert audio to 16-bit.
	final_signal = (cipher_signal * 32768).astype('int16')

	# Export to WAV file.
	out_file = textfile.split(".")[0] + ".wav"
	scipy.io.wavfile.write(out_file, SAMPLE_RATE, final_signal)





