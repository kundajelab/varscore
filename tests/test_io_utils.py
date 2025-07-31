from varscore.utils.io_utils import load_peaks, NARROWPEAK_SCHEMA

SAMPLE_PEAKS_FILE_WITH_SUMMIT = "tests/data/sample_beds/with_summits.narrowPeak"
SAMPLE_PEAKS_FILE_NO_SUMMIT = "tests/data/sample_beds/without_summits_bed3.bed"

def test_load_peaks_with_summit_offset():
    """Test load_peaks function with provided summit_offset values."""
    width = 2114
    result_df = load_peaks(SAMPLE_PEAKS_FILE_WITH_SUMMIT, width=width)
    
    # Check that all expected columns are present
    expected_columns = NARROWPEAK_SCHEMA + ["summit_pos", "window_start", "window_end"]
    assert all(col in result_df.columns for col in expected_columns)
    
    # Check correct number of rows
    assert len(result_df) == 2
    
    # Check specific values for first row
    # chr1	100004111	100004627	.	920	.	15.02412	92.06185	89.55358	337
    flank_size = width // 2  # 1057
    expected_summit_pos = 100004111 + 337  # start + summit_offset
    expected_window_start = expected_summit_pos - flank_size  # 100004111 - 1057 = 100003054
    expected_window_end = expected_summit_pos + flank_size  # 100004111 + 1057 = 100005168
    
    assert result_df.iloc[0]["summit_pos"] == expected_summit_pos
    assert result_df.iloc[0]["window_start"] == expected_window_start
    assert result_df.iloc[0]["window_end"] == expected_window_end


def test_load_peaks_without_summit_offset():
    """Test load_peaks function when summit_offset is NaN and needs to be calculated."""
    width = 2114
    result_df = load_peaks(SAMPLE_PEAKS_FILE_NO_SUMMIT, width=width)
    assert (result_df['summit_offset'] == 500).all()


def test_load_peaks_different_width():
    """Test load_peaks function with different width values."""
    width = 1000
    result_df = load_peaks(SAMPLE_PEAKS_FILE_WITH_SUMMIT, width=width)
    
    flank_size = width // 2  # 500
    expected_summit_pos = 100004111 + 337  # start + summit_offset for first row
    expected_window_start = expected_summit_pos - flank_size  # 100004111 - 500 = 100003611
    expected_window_end = expected_summit_pos + flank_size  # 100004111 + 500 = 100004611
    
    assert result_df.iloc[0]["window_start"] == expected_window_start
    assert result_df.iloc[0]["window_end"] == expected_window_end
