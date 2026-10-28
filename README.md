# snowwi-tools

## Optional GPU range compression

`snowwi_tools.ra.gpu_range_compression` provides a CuPy backend for the
existing SNOWWI pulse-processing chain: the same forward/backward SOS
bandpass filter, optional downconversion/lowpass filter, Hamming chirp
replica, and linear correlation with SciPy's `mode='same'` alignment.
Sample rate, IF carrier, phase convention and amplitude scaling are retained.
Intermediates use double precision; output can be complex64 or complex128.

Install one optional extra matching the CUDA environment (not both):

```bash
python -m pip install -e '.[gpu-cu11]'   # CUDA 11.x toolkit, including 11.8
# python -m pip install -e '.[gpu-cu12]' # CUDA 12.x environment
```

CPU use does not import or require CuPy. The GPU backend is opt-in; a missing
CuPy installation or CUDA failure raises an error instead of silently
switching backends. It was developed against CuPy 13.6.

The `GpuRangeCompressor` class accepts batches of raw pulse rows. The
`read_and_compress_gpu` helper removes file headers, reads only the requested
pulses, truncates the raw swath before filtering, and skips initial samples
after compression. Device memory is bounded by the pulse batch size; the
returned patch is a NumPy array for the existing gangler focuser.
The optional `scale_factor` argument applies the existing raw-data scaling
before filtering; it defaults to 1.0. Gangler supplies its `Scaling factor`
configuration value to this argument.

In gangler's `[Azimuth Compression]` configuration, select:

```ini
Range Compression Backend = cupy
GPU Range Batch Size = 128
```

The default is `cpu`. Start with a small isolated patch and compare complex
samples with the CPU product before using a new processing configuration.
Changing `Last Swath Sample` changes filtering boundaries; it is not a
performance-only option and should be held fixed for comparisons.

GPU numerical tests, including all bands, even/odd replicas and file slicing:

```bash
python -m pytest -q snowwi_tools/tests/test_gpu_range_compression.py
```

These tests require a visible CUDA device. CuPy runs GPU FFTs through cuFFT;
see the [CuPy correlation documentation](https://docs.cupy.dev/en/v13.6.0/reference/generated/cupyx.scipy.signal.correlate.html).
