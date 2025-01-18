from deeplift.dinuc_shuffle import dinuc_shuffle
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
import tensorflow_probability as tfp


# THESE ARE ALL FUNCTIONS TAKEN FROM THE CHROMBPNET REPO

###
# SEQUENCE FUNCTIONS
###
def dna_to_one_hot(seqs):
	"""
	TAKEN FROM CHROMBPNET.TRAINING.UTILS.ONE_HOT.PY
	Converts a list of DNA ("ACGT") sequences to one-hot encodings, where the
	position of 1s is ordered alphabetically by "ACGT". `seqs` must be a list
	of N strings, where every string is the same length L. Returns an N x L x 4
	NumPy array of one-hot encodings, in the same order as the input sequences.
	All bases will be converted to upper-case prior to performing the encoding.
	Any bases that are not "ACGT" will be given an encoding of all 0s.
	"""
	seq_len = len(seqs[0])
	assert np.all(np.array([len(s) for s in seqs]) == seq_len)
	# Join all sequences together into one long string, all uppercase
	seq_concat = "".join(seqs).upper() + "ACGT"
	# Add one example of each base, so np.unique doesn't miss indices later
	one_hot_map = np.identity(5)[:, :-1].astype(np.int8)
	# Convert string into array of ASCII character codes;
	base_vals = np.frombuffer(bytearray(seq_concat, "utf8"), dtype=np.int8)
	# Anything that's not an A, C, G, or T gets assigned a higher code
	base_vals[~np.isin(base_vals, np.array([65, 67, 71, 84]))] = 85
	# Convert the codes into indices in [0, 4], in ascending order by code
	_, base_inds = np.unique(base_vals, return_inverse=True)
	# Get the one-hot encoding for those indices, and reshape back to separate
	return one_hot_map[base_inds[:-4]].reshape((len(seqs), seq_len, 4))	


###
# MODEL FUNCTIONS
###
def multinomial_nll(true_counts, logits):
	"""
	TAKEN FROM CHROMBPNET.TRAINING.UTILS.LOSSES.PY
	Compute the multinomial negative log-likelihood
	Args:
	true_counts: observed count values
	logits: predicted logit values
	"""
	counts_per_example = tf.reduce_sum(true_counts, axis=-1)
	dist = tfp.distributions.Multinomial(total_count=counts_per_example, logits=logits)
	return (-tf.reduce_sum(dist.log_prob(true_counts)) / tf.cast(tf.shape(true_counts)[0], dtype=tf.float32))


###
# SHAP FUNCTIONS
###
def shuffle_several_times(s, numshuffles=20):
	"""
	TAKEN FROM CHROMBPNET.EVALUATION.INTERPRET.SHAP_UTILS.PY
	"""
	if len(s) == 2:
		return [np.array([dinuc_shuffle(s[0]) for i in range(numshuffles)]), np.array([s[1] for i in range(numshuffles)])]
	else:
		return [np.array([dinuc_shuffle(s[0]) for i in range(numshuffles)])]


def combine_mult_and_diffref(mult, orig_inp, bg_data):
	"""
	TAKEN FROM CHROMBPNET.EVALUATION.INTERPRET.SHAP_UTILS.PY
	"""
	to_return = []

	for l in [0]:
		projected_hypothetical_contribs = np.zeros_like(bg_data[l]).astype("float")
		assert (len(orig_inp[l].shape) == 2)
        
		# At each position in the input sequence, we iterate over the
		# one-hot encoding possibilities (eg: for genomic sequence, 
		# this is ACGT i.e. 1000, 0100, 0010 and 0001) and compute the
		# hypothetical difference-from-reference in each case. We then 
		# multiply the hypothetical differences-from-reference with 
		# the multipliers to get the hypothetical contributions. For 
		# each of the one-hot encoding possibilities, the hypothetical
		# contributions are then summed across the ACGT axis to 
		# estimate the total hypothetical contribution of each 
		# position. This per-position hypothetical contribution is then
		# assigned ("projected") onto whichever base was present in the
		# hypothetical sequence. The reason this is a fast estimate of
		# what the importance scores *would* look like if different 
		# bases were present in the underlying sequence is that the
		# multipliers are computed once using the original sequence, 
		# and are not computed again for each hypothetical sequence.
		for i in range(orig_inp[l].shape[-1]):
			hypothetical_input = np.zeros_like(orig_inp[l]).astype("float")
			hypothetical_input[:, i] = 1.0
			hypothetical_difference_from_reference = (hypothetical_input[None, :, :] - bg_data[l])
			hypothetical_contribs = hypothetical_difference_from_reference * mult[l]
			projected_hypothetical_contribs[:, :, i] = np.sum(hypothetical_contribs, axis=-1)
            
		to_return.append(np.mean(projected_hypothetical_contribs,axis=0))

	if len(orig_inp) > 1:
		to_return.append(np.zeros_like(orig_inp[1]))
    
	return to_return

