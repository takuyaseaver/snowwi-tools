"""Optional, bounded-memory GPU backend for the existing SNOWWI signal chain.

CuPy is imported only when this backend is selected. Filtering uses the same
SciPy-designed SOS coefficients and odd extension as the CPU implementation.
Correlation retains the legacy IF/baseband, replica and ``mode='same'``
conventions and configured raw scaling; no decimation or timing correction is added.
"""
import operator
from pathlib import Path

import numpy as np
from scipy.fft import next_fast_len

from snowwi_tools.lib.signal_processing import (
    butter_bandpass, butter_lowpass, exp_chirp,
)


class _ForwardBackwardSos:
    """Cache CuPy's steady-state filter conditions across pulse batches."""

    def __init__(self, cp, coefficients):
        from cupyx.scipy.signal import sosfilt, sosfilt_zi
        self.cp = cp
        self.sosfilt = sosfilt
        self.sos = cp.asarray(coefficients)
        self.edge = 3 * (2 * len(coefficients) + 1 - min(
            np.count_nonzero(coefficients[:, 2] == 0),
            np.count_nonzero(coefficients[:, 5] == 0)))
        # CuPy's SOS implementation has four states per section. Its zi is
        # intentionally computed with CuPy, not scipy.signal.sosfilt_zi.
        self.zi = sosfilt_zi(self.sos)[:, None, :]

    def __call__(self, data):
        if data.shape[1] <= self.edge:
            raise ValueError(f'The input length must exceed the filter padlen {self.edge}')
        n = self.edge
        cp = self.cp
        extended = cp.concatenate((
            2 * data[:, :1] - data[:, n:0:-1], data,
            2 * data[:, -1:] - data[:, -2:-n-2:-1]), axis=-1)
        forward, _ = self.sosfilt(
            self.sos, extended, axis=-1,
            zi=self.zi * extended[:, 0][None, :, None])
        backward, _ = self.sosfilt(
            self.sos, forward[:, ::-1], axis=-1,
            zi=self.zi * forward[:, -1][None, :, None])
        return backward[:, ::-1][:, n:-n]


class GpuRangeCompressor:
    """Reuse filter coefficients and replica spectra across pulse batches.

    ``precision`` controls the returned complex array, as in ``compress``.
    Internal filtering and FFTs use float64/complex128 to match the CPU path
    even when its final output is complex64. Call from the processing thread,
    not from the CPU file-reader thread pool.
    """

    def __init__(self, chirp, band_filter=None, window='hamming',
                 precision='double', do_downconvert=False, scale_factor=1.0):
        if precision.lower() not in ('single', 'double'):
            raise ValueError("precision must be 'single' or 'double'")
        try:
            import cupy as cp
            from cupyx.scipy.signal import sosfilt, sosfilt_zi
        except ImportError as exc:
            raise RuntimeError(
                "The cupy range backend requires CuPy with cupyx.scipy.signal. "
                "Install a CuPy build compatible with your CUDA environment, "
                "or select 'Range Compression Backend = cpu'."
            ) from exc
        self.cp = cp
        self.dtype = np.complex64 if precision.lower() == 'single' else np.complex128
        self.fs = chirp['fs']
        self.fc = (chirp['f_h'] + chirp['f_l']) / 2
        self.do_downconvert = do_downconvert
        self.scale_factor = float(scale_factor)
        self.band_sos = None
        if band_filter is not None:
            self.band_sos = _ForwardBackwardSos(cp, butter_bandpass(
                band_filter['lowcut'], band_filter['highcut'], band_filter['fs']))
        f_l, f_h = chirp['f_l'], chirp['f_h']
        self.lowpass_sos = None
        if do_downconvert:
            bw = f_h - f_l
            self.lowpass_sos = _ForwardBackwardSos(cp, butter_lowpass(bw / 2, self.fs))
            f_l, f_h = -bw / 2, bw / 2
        # These are intentionally the legacy np.linspace endpoint and int()
        # length, including exp_chirp's conjugate/reverse down-chirp definition.
        n = int(self.fs * chirp['tp'])
        if n < 1:
            raise ValueError('The reference chirp must contain at least one sample')
        t = np.linspace(0, chirp['tp'], n)
        ref = exp_chirp(t, f_l, chirp['tp'], f_h, chirp['chirp_type'])
        if window == 'hanning':
            ref *= np.hanning(n)
        elif window == 'hamming':
            ref *= np.hamming(n)
        self.replica_length = n
        self.kernel = cp.asarray(ref[::-1].conj())
        self._spectra = {}
        self._mixers = {}

    def compress(self, rows, skip_samples=0):
        """Filter and compress a 2-D host batch; return its host complex array."""
        if np.ndim(rows) != 2 or rows.shape[1] == 0:
            raise ValueError('rows must be a nonempty-width 2-D pulse array')
        skip_samples = operator.index(skip_samples)
        n_samples = rows.shape[1]
        if not 0 <= skip_samples <= n_samples:
            raise ValueError('skip_samples must be within the input row')
        cp = self.cp
        # scipy.sosfiltfilt promotes int16 raw samples to float64.
        data = cp.asarray(rows, dtype=cp.float64)
        if self.scale_factor != 1.0:
            data *= self.scale_factor
        if self.band_sos is not None:
            data = self.band_sos(data)
        if self.do_downconvert:
            if n_samples not in self._mixers:
                t = np.arange(n_samples) / self.fs
                # downconvert() defaults to a full-row Hamming window and
                # complex128 intermediates, independently of output precision.
                lo = np.exp(-2j * np.pi * self.fc * t) * np.hamming(n_samples)
                self._mixers[n_samples] = cp.asarray(lo)
            data = self.lowpass_sos(data * self._mixers[n_samples])
        n_fft = next_fast_len(n_samples + self.replica_length - 1)
        if n_fft not in self._spectra:
            self._spectra[n_fft] = cp.fft.fft(self.kernel, n=n_fft)
        spectrum = cp.fft.fft(data, n=n_fft, axis=-1)
        spectrum *= self._spectra[n_fft]
        full = cp.fft.ifft(spectrum, axis=-1)
        # Linear convolution with conjugate(reverse(ref)), centred exactly
        # like scipy.signal.correlate(rows[i], ref, mode='same'). For even
        # replicas this preserves the peak's len(ref)//2 sample offset.
        start = (self.replica_length - 1) // 2
        out = full[:, start + skip_samples:start + n_samples]
        return cp.asnumpy(out.astype(self.dtype, copy=False))


def read_and_compress_gpu(filelist, data_samples, header_samples, last_sample,
                          skip_samples, chirp, band_filter=None,
                          from_pulse=0, to_pulse=None, precision='double',
                          do_downconvert=False, batch_size=128, scale_factor=1.0):
    """Read only requested file rows, process on GPU, and return ``(rc, width)``.

    Bounds use normal Python slicing over the concatenated input files.
    ``last_sample`` truncates raw rows before filtering; ``skip_samples`` is
    applied after compression, matching read_and_compress_local. GPU memory
    is bounded by batch_size, while the requested patch remains in host RAM
    for the existing PyCUDA focuser. Files are never modified.
    """
    batch_size = operator.index(batch_size)
    if batch_size < 1:
        raise ValueError('GPU Range Batch Size must be positive')
    data_samples = operator.index(data_samples)
    header_samples = operator.index(header_samples)
    last_sample = operator.index(last_sample)
    skip_samples = operator.index(skip_samples)
    if data_samples < 1 or header_samples < 0:
        raise ValueError('Invalid raw record dimensions')
    if not 0 < last_sample <= data_samples:
        raise ValueError('Last Swath Sample must be within the raw row')
    if not 0 <= skip_samples <= last_sample:
        raise ValueError('Skip Samples must be within the retained swath')
    stride = data_samples + header_samples
    files = []
    for name in filelist:
        size = Path(name).stat().st_size
        count, remainder = divmod(size, stride * np.dtype(np.int16).itemsize)
        if remainder:
            raise ValueError(f'Incomplete raw record in {name}')
        files.append((name, count))
    total_rows = sum(count for _, count in files)
    first, stop, _ = slice(from_pulse, to_pulse).indices(total_rows)
    n_rows = max(0, stop - first)
    width = last_sample - skip_samples
    if precision.lower() not in ('single', 'double'):
        raise ValueError("precision must be 'single' or 'double'")
    dtype = np.complex64 if precision.lower() == 'single' else np.complex128
    if n_rows == 0:
        return np.empty((0, width), dtype=dtype), width
    compressor = GpuRangeCompressor(
        chirp, band_filter, precision=precision, do_downconvert=do_downconvert,
        scale_factor=scale_factor)
    result = np.empty((n_rows, width), dtype=dtype)
    file_start = 0
    output_start = 0
    for name, count in files:
        local_first = max(0, first - file_start)
        local_stop = min(count, stop - file_start)
        file_start += count
        if local_stop <= local_first:
            continue
        raw = np.memmap(name, dtype=np.int16, mode='r', shape=(count, stride))
        try:
            for begin in range(local_first, local_stop, batch_size):
                end = min(begin + batch_size, local_stop)
                rows = raw[begin:end, header_samples:header_samples + last_sample]
                result[output_start:output_start + end - begin] = compressor.compress(
                    rows, skip_samples=skip_samples)
                output_start += end - begin
                del rows
        finally:
            # Only host results escape; GPU work has completed in asnumpy().
            del raw
    return result, width
