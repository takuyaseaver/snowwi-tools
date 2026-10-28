"""Numerical contract for the optional GPU range backend (requires a GPU)."""
import numpy as np
import pytest
import sys

from snowwi_tools.lib.file_handling import read_and_compress_local
from snowwi_tools.lib.signal_processing import butter_bandpass_filter, exp_chirp
from snowwi_tools.params import get_band_params_4x2
from snowwi_tools.ra.gpu_range_compression import GpuRangeCompressor, read_and_compress_gpu
from snowwi_tools.ra.radar_processing import compress, downconvert


@pytest.fixture(scope='module')
def gpu():
    cp = pytest.importorskip('cupy')
    # Other suites may have created PyCUDA's private focuser context first.
    # Standalone CuPy tests must use its primary context consistently with
    # kernels cached by the gangler integration tests in the same process.
    driver = sys.modules.get('pycuda.driver')
    context = None
    try:
        if driver is not None and driver.Context.get_current() is not None:
            context = driver.Context.get_device().retain_primary_context()
            context.push()
        try:
            cp.cuda.Device().compute_capability
        except cp.cuda.runtime.CUDARuntimeError as exc:
            pytest.skip(f'CUDA device unavailable: {exc}')
        yield cp
    finally:
        if context is not None:
            context.detach()


def cpu_reference(rows, chirp, filt, precision, baseband):
    data = butter_bandpass_filter(rows, filt['lowcut'], filt['highcut'], filt['fs'])
    lo, hi = chirp['f_l'], chirp['f_h']
    if baseband:
        data = downconvert(data, hi, lo, chirp['fs'])
        lo, hi = -(hi-lo)/2, (hi-lo)/2
    return compress(data, hi, lo, chirp['tp'], chirp['fs'],
                    type=chirp['chirp_type'], precision=precision)


@pytest.mark.parametrize('band', ['low', 'high', 'c'])
@pytest.mark.parametrize('precision', ['single', 'double'])
@pytest.mark.parametrize('baseband', [False, True])
def test_complex_samples_match_cpu(gpu, band, precision, baseband):
    params = get_band_params_4x2(band)
    chirp = dict(f_h=params['f_h'], f_l=params['f_l'], tp=14e-6,
                 fs=1.2288e9, chirp_type=params['chirp_type'])
    filt = dict(lowcut=params['lowcut'], highcut=params['highcut'], fs=chirp['fs'])
    rows = np.random.default_rng(12).integers(-1000, 1000, (3, 25000), dtype=np.int16)
    rows[:, 0] = 15000
    rows[:, -1] = -10000  # include filter transients at both record edges
    expected = cpu_reference(rows, chirp, filt, precision, baseband)
    actual = GpuRangeCompressor(chirp, filt, precision=precision,
                                do_downconvert=baseband).compress(rows)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    # Relative L2 controls the complex signal, including phase, rather than
    # magnitude-only agreement or relative error at near-zero samples.
    error = np.linalg.norm(actual-expected) / np.linalg.norm(expected)
    assert error < (2e-6 if precision == 'single' else 2e-9)
    np.testing.assert_allclose(actual, expected, rtol=0,
                               atol=np.max(abs(expected)) * (2e-6 if precision == 'single' else 2e-9))


@pytest.mark.parametrize('length', [128, 129])
def test_same_mode_peak_alignment(gpu, length):
    fs = 64e6
    chirp = dict(f_l=10e6, f_h=20e6, tp=length/fs, fs=fs, chirp_type='up')
    ref = exp_chirp(np.linspace(0, chirp['tp'], length),
                   chirp['f_l'], chirp['tp'], chirp['f_h'], 'up')
    data = np.zeros((1, 1024))
    data[0, 200:200+length] = ref.real
    expected = compress(data, 20e6, 10e6, chirp['tp'], fs, type='up')
    actual = GpuRangeCompressor(chirp).compress(data)
    assert np.argmax(abs(actual)) == np.argmax(abs(expected)) == 200+length//2
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-11)


@pytest.mark.parametrize('scale_factor', [1.0, 0.125, -2.0])
def test_file_slices_headers_truncation_and_partial_batches(gpu, tmp_path, scale_factor):
    chirp = dict(f_l=10e6, f_h=20e6, tp=2e-6, fs=64e6, chirp_type='up')
    filt = dict(lowcut=8e6, highcut=22e6, fs=64e6)
    files = []
    expected = []
    for i, count in enumerate([5, 5, 3]):
        raw = np.random.default_rng(i).integers(-1000, 1000, (count, 1028), dtype=np.int16)
        raw[:, :4] = 30000  # make accidental header inclusion visible
        path = tmp_path/f'stream0_{i}.dat'
        raw.tofile(path)
        files.append(path)
        expected.append(read_and_compress_local(
            str(path), 1024, 4, 11, 900, chirp, 'hamming', filt, True,
            scale_factor=scale_factor))
    expected = np.concatenate(expected)[2:12]
    actual, width = read_and_compress_gpu(
        files, 1024, 4, 900, 11, chirp, filt,
        from_pulse=2, to_pulse=12, batch_size=3, scale_factor=scale_factor)
    assert actual.shape == (10, 889)
    assert width == 889
    np.testing.assert_allclose(actual, expected, rtol=0, atol=np.max(abs(expected))*2e-9)


def test_invalid_batch_and_incomplete_record(tmp_path):
    with pytest.raises(ValueError, match='positive'):
        read_and_compress_gpu([], 1024, 4, 1024, 0, {}, batch_size=0)
    path = tmp_path/'incomplete.dat'
    path.write_bytes(b'123')
    with pytest.raises(ValueError, match='Incomplete raw record'):
        read_and_compress_gpu([path], 1024, 4, 1024, 0, {})


def test_empty_selection_does_not_require_cuda(tmp_path):
    path = tmp_path/'stream0_0.dat'
    np.zeros((3, 1028), dtype=np.int16).tofile(path)
    output, width = read_and_compress_gpu(
        [path], 1024, 4, 1024, 0, {}, from_pulse=2, to_pulse=1)
    assert output.shape == (0, 1024)
    assert width == 1024
