import modal
from deeplift.dinuc_shuffle import dinuc_shuffle
import numpy as np
import pandas as pd
from typing import Tuple
import pyfaidx
import shap
import tensorflow as tf
import itertools
from typing import Optional
from dataclasses import dataclass
from enum import Enum
from scipy.spatial.distance import jensenshannon
from scipy.stats import binom
from scipy.special import softmax

tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
import tensorflow_probability as tfp

class ValidationErrorReason(Enum):
    """Reasons why a variant validation might fail."""
    REF_MISMATCH = "reference_mismatch"
    ALT_LENGTH_MISMATCH = "alt_length_mismatch"
    ALT_MISMATCH = "alt_mismatch"
    GENOME_ACCESS_ERROR = "genome_access_error"


@dataclass
class ValidationResult:
    """Result of validating a single variant."""
    is_valid: bool
    ref_seq: Optional[str] = None
    alt_seq: Optional[str] = None
    error_reason: Optional[ValidationErrorReason] = None
    error_message: Optional[str] = None

app = modal.App("chrombpnet-model")
vol = modal.Volume.from_name("model-cache")
image = modal.Image.from_registry("kundajelab/chrombpnet:latest")

# Module-level caches
_model_cache = {}
_peak_distr_cache = {}

@app.function(gpu="H100", image=image, volumes={"/data": vol}, timeout=60*20)
def score_variant_batch(variants: pd.DataFrame, model_loc: str, peak_distr_loc: str, genome_loc: str):
    global _model_cache, _peak_distr_cache
  
    if model_loc not in _model_cache:
        print(f"Loading model {model_loc}...")
        _model_cache[model_loc] = load_chrombpnet(model_loc)
    if peak_distr_loc not in _peak_distr_cache:
        _peak_distr_cache[peak_distr_loc] = np.load(peak_distr_loc)
    
    model = _model_cache[model_loc]
    peaks_dist = _peak_distr_cache[peak_distr_loc]
    genome = pyfaidx.Fasta(genome_loc)

    ref_seqs, alt_seqs = get_variant_seqs_with_genome(variants, genome)
    ref_pred_logits, ref_pred_logcts = predict(model, ref_seqs)
    alt_pred_logits, alt_pred_logcts = predict(model, alt_seqs)
    
    print(f"Computing scores...")
    result = _score_variant_df(
        variants,
        ref_pred_logits,
        alt_pred_logits,
        ref_pred_logcts,
        alt_pred_logcts,
        peaks_dist,
    )
    
    print(f"Scores computed successfully")
    print(result.head())
    
    return result


@app.function(image=image, volumes={"/data": vol}, timeout=60*60*3)
def score_variants(model_loc: str, peak_distr_loc: str, variants_loc: str, genome_loc: str, batch_size: int = 250_000):
    
    print("Loading variants...")
    variants = load_variants(variants_loc)
    print(f"Variants loaded successfully: {len(variants)} variants")
    
    # TEST LIMIT
    # variants = variants.iloc[:100]
    
    batches = [variants.iloc[i:i+batch_size] for i in range(0, len(variants), batch_size)]
    scored_batches = list(
      score_variant_batch.map(batches, itertools.repeat(model_loc), itertools.repeat(peak_distr_loc), itertools.repeat(genome_loc))
    )
    
    print(f"Scored {len(scored_batches)} batches")
    print(scored_batches[0].head())
    print(f"Total variants scored: {sum(len(batch) for batch in scored_batches)}")

    return scored_batches

def load_chrombpnet(model_loc: str) -> tf.keras.Model:
    """Loads a ChromBPNet model."""
    custom_objects = {"multinomial_nll": _multinomial_nll, "tf": tf}
    get_custom_objects().update(custom_objects)
    model = load_model(model_loc, compile=False)
    return model
  
VARIANT_SCHEMA = ["chr", "pos", "ref", "alt", "variant_id"]


def load_variants(variants_loc: str) -> pd.DataFrame:
    """Load a variants DataFrame."""
    variants_df = pd.read_csv(variants_loc, sep="\t", names=VARIANT_SCHEMA)
    return variants_df
  
def get_variant_seqs_with_genome(
    variants_df: pd.DataFrame, genome: pyfaidx.Fasta, width: int = 2114
) -> Tuple[np.ndarray, np.ndarray]:
    """Get one-hot encoded ref/alt sequences from variants using pre-opened genome.
    
    This function raises AssertionError for invalid variants (legacy behavior).
    """
    ref_sequences = []
    alt_sequences = []
    
    for _, row in variants_df.iterrows():
        chro, pos, ref, alt = (
            row["chr"],
            int(row["pos"]) - 1,
            row["ref"],
            row["alt"],
        )
        
        result = validate_variant(chro, pos, ref, alt, genome, width)
        
        if not result.is_valid:
            full_error = f"Failed to get sequence for variant {chro}:{pos}:{ref}:{alt} : {result.error_reason.value} - {result.error_message}"
            print(full_error)
            raise AssertionError(full_error)
        
        ref_sequences.append(result.ref_seq)
        alt_sequences.append(result.alt_seq)
    
    # Convert to one-hot encoding
    ref_onehot = dna_to_one_hot(ref_sequences)
    alt_onehot = dna_to_one_hot(alt_sequences)
    return ref_onehot, alt_onehot

  
def predict(
    model: tf.keras.Model, seqs: np.ndarray, batch_size: int = 512
) -> Tuple[np.ndarray, np.ndarray]:
    """Make predictions on sequences."""
    pred_logits_batches, pred_logcts_batches = [], []
    for i in range(0, seqs.shape[0], batch_size):
        seq_batch = seqs[i : i + batch_size]
        pred_logits_i, pred_logcts_i = model.predict_on_batch(seq_batch)
        pred_logits_batches.append(pred_logits_i)
        pred_logcts_batches.append(pred_logcts_i)
    pred_logits = np.vstack(pred_logits_batches)
    pred_logcts = np.vstack(pred_logcts_batches)
    return pred_logits, pred_logcts


def deepshap(model: tf.keras.Model, seqs: np.ndarray) -> np.ndarray:
    """Compute counts DeepSHAPs on sequences."""
    counts_explainer = shap.explainers.deep.TFDeepExplainer(
        (model.input, tf.reduce_sum(model.outputs[1], axis=-1)),
        shuffle_several_times,
        combine_mult_and_diffref=combine_mult_and_diffref,
    )
    counts_deepshaps = counts_explainer.shap_values(seqs, progress_message=100)
    observed_counts_deepshaps = (seqs * counts_deepshaps).astype(np.float16)
    return observed_counts_deepshaps


def _multinomial_nll(true_counts, logits):
    """
    TAKEN FROM CHROMBPNET.TRAINING.UTILS.LOSSES.PY

    Compute the multinomial negative log-likelihood
    Args:
    true_counts: observed count values
    logits: predicted logit values
    """
    counts_per_example = tf.reduce_sum(true_counts, axis=-1)
    dist = tfp.distributions.Multinomial(total_count=counts_per_example, logits=logits)
    return -tf.reduce_sum(dist.log_prob(true_counts)) / tf.cast(
        tf.shape(true_counts)[0], dtype=tf.float32
    )


def model_is_valid(model: tf.keras.Model) -> bool:
    """Checks that a ChromBPNet model is valid."""
    # TODO: IMPLEMENT (Not sure what to do for this, maybe check input size?)
    return True


#######################
# SEQUENCE GENERATION #
#######################
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


##################
# SHAP FUNCTIONS #
##################
def shuffle_several_times(s, numshuffles=20):
    """
    TAKEN FROM CHROMBPNET.EVALUATION.INTERPRET.SHAP_UTILS.PY
    """

    if len(s) == 2:
        return [
            np.array([dinuc_shuffle(s[0]) for i in range(numshuffles)]),
            np.array([s[1] for i in range(numshuffles)]),
        ]
    else:
        return [np.array([dinuc_shuffle(s[0]) for i in range(numshuffles)])]


def combine_mult_and_diffref(mult, orig_inp, bg_data):
    """
    TAKEN FROM CHROMBPNET.EVALUATION.INTERPRET.SHAP_UTILS.PY
    """
    to_return = []

    for l in [0]:
        projected_hypothetical_contribs = np.zeros_like(bg_data[l]).astype("float")
        assert len(orig_inp[l].shape) == 2

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
            hypothetical_difference_from_reference = (
                hypothetical_input[None, :, :] - bg_data[l]
            )
            hypothetical_contribs = hypothetical_difference_from_reference * mult[l]
            projected_hypothetical_contribs[:, :, i] = np.sum(
                hypothetical_contribs, axis=-1
            )

        to_return.append(np.mean(projected_hypothetical_contribs, axis=0))

    if len(orig_inp) > 1:
        to_return.append(np.zeros_like(orig_inp[1]))

    return to_return

def validate_variant(
    chro: str, pos: int, ref: str, alt: str, genome: pyfaidx.Fasta, width: int = 2114
) -> ValidationResult:
    """Validate a single variant.
    
    Args:
        chro: Chromosome
        pos: Position (0-based)
        ref: Reference allele
        alt: Alternate allele
        genome: Open genome file handle
        width: Sequence width
        
    Returns:
        ValidationResult with validation status and error details if invalid
    """
    try:
        ref_seq = str(genome[chro][pos - width // 2 : pos + width // 2])
        
        if ref_seq[width // 2 : width // 2 + len(ref)] != ref:
            return ValidationResult(
                is_valid=False,
                error_reason=ValidationErrorReason.REF_MISMATCH,
                error_message=f"Expected {ref}, got {ref_seq[width // 2 : width // 2 + len(ref)]}"
            )
        
        alt_seq = (
            ref_seq[: width // 2]
            + alt
            + str(genome[chro][pos + len(ref) : pos + width // 2 + len(ref) - len(alt)])
        )
        
        if len(alt_seq) != width:
            return ValidationResult(
                is_valid=False,
                error_reason=ValidationErrorReason.ALT_LENGTH_MISMATCH,
                error_message=f"Expected length {width}, got {len(alt_seq)}"
            )
        
        if alt_seq[width // 2 : width // 2 + len(alt)] != alt:
            return ValidationResult(
                is_valid=False,
                error_reason=ValidationErrorReason.ALT_MISMATCH,
                error_message=f"Expected {alt}, got {alt_seq[width // 2 : width // 2 + len(alt)]}"
            )
        
        return ValidationResult(
            is_valid=True,
            ref_seq=ref_seq,
            alt_seq=alt_seq
        )
        
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            error_reason=ValidationErrorReason.GENOME_ACCESS_ERROR,
            error_message=str(e)
        )
        
def _score_variant_df(
    variant_df,
    ref_pred_logits,
    alt_pred_logits,
    ref_pred_logcts,
    alt_pred_logcts,
    peaks_dist,
):
    variant_df["lfc"] = _compute_lfc(ref_pred_logcts, alt_pred_logcts)
    # TODO: test this more before enabling; gives NaNs somewhat often?
    # variant_df["lfc_pval"] = _compute_lfc_pval(ref_pred_logcts, alt_pred_logcts)
    variant_df["jsd"] = _compute_jsd(ref_pred_logits, alt_pred_logits)
    variant_df["active_allele_quantile"] = _compute_active_allele_quantile(
        ref_pred_logcts, alt_pred_logcts, peaks_dist
    )
    variant_df["ips"] = _compute_ips(
        variant_df["lfc"], variant_df["jsd"], variant_df["active_allele_quantile"]
    )
    return variant_df


def _compute_lfc(ref_logcts, alt_logcts):
    """Computes LFC.

    Args:
        ref_logcts: Reference allele logcounts. Shape: (N, 1).
        alt_logcts: Alternate allele logcounts. Shape: (N, 1).
    """
    return (alt_logcts - ref_logcts) / np.log(2)


def _compute_lfc_pval(ref_logcts, alt_logcts):
    """Computes LFC p-values.

    Args:
        ref_logcts: Reference allele logcounts. Shape: (N, 1).
        alt_logcts: Alternate allele logcounts. Shape: (N, 1).
    """
    min_cts = np.exp(np.minimum(ref_logcts, alt_logcts))
    total_cts = np.exp(ref_logcts) + np.exp(alt_logcts)
    return [binom.cdf(min_cts[i], total_cts[i], 0.5) for i in range(len(ref_logcts))]


def _compute_jsd(ref_logits, alt_logits):
    """Computes JSD.

    Args:
        ref_logits: Reference allele logits. Shape: (N, 1000).
        alt_logits: Alternate allele logits. Shape: (N, 1000).
    """
    ref_profile = softmax(ref_logits, axis=1)
    alt_profile = softmax(alt_logits, axis=1)
    return np.squeeze(
        [jensenshannon(x, y, base=2.0) for x, y in zip(ref_profile, alt_profile)]
    )


def _compute_active_allele_quantile(ref_logcts, alt_logcts, peaks_dist):
    """
    Computes the active allele quantile.

    Args:
        ref_logcts: Reference allele logcounts. Shape: (N, 1).
        alt_logcts: Alternate allele logcounts. Shape: (N, 1).
        peaks_dist: Peak distribution. Shape: (1000,).

    Returns:
        active_allele_quantile: Active allele quantile. Shape: (N, 1).
    """
    ref_quantiles = np.searchsorted(peaks_dist, ref_logcts) / len(peaks_dist)
    alt_quantiles = np.searchsorted(peaks_dist, alt_logcts) / len(peaks_dist)
    return np.maximum(ref_quantiles, alt_quantiles)


def _compute_ips(lfc, jsd, active_allele_quantile):
    return lfc * jsd * active_allele_quantile


@app.local_entrypoint()
def main():
    score_variants.remote("/data/Brain_c0__fold_0__chrombpnet_nobias.h5", "/data/Brain_c0_peak_distr.npy", "/data/variant-5M.tsv", "/data/genome.fasta")
