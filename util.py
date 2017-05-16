"""
Utility functions for audio processing, etc.
"""
import string

def millis_to_samples(sample_rate, ms):
	""" Convert a millisecond duration to a length in samples. """
	return int(sample_rate * (ms/1000))

def samples_to_millis(sample_rate, n_samples):
	""" Convert a sample count to millisecond duration. """
	return (n_samples/sample_rate)*1000.0

def letter_pos(l):
	""" Letter position in alphabet. SPACE is 27th letter. """
	s = string.ascii_lowercase+" "
	return s.index(l)

def pos_to_letter(i):
	""" Convert alphabet position to letter. SPACE is 27th letter. """
	return (string.ascii_lowercase+" ")[i]
