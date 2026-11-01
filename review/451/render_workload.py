"""Run actual room RIR construction and recorded-speech rendering.

Only the distributed WAV is read. No dataset code is imported or executed.
The native extension and every array's provenance are recorded.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
source = a.source.resolve()
sys.path.insert(0, str(source))
import pyroomacoustics as pra

assert Path(pra.__file__).resolve().parent.parent == source
assert Path(pra.libroom.__file__).resolve().parent.parent == source
wav = source/'examples/input_samples/cmu_arctic_us_aew_a0001.wav'
speech, fs = sf.read(wav)
assert fs == 16000 and speech.ndim == 1
speech = speech[:fs]
mics = np.array([[1.6, 3., .9], [3., 1.6, 1.8], [4.4, 3., .9]]).T
arrays, records = {}, []
for mode, order in [('hybrid', k) for k in range(4)]+[('rt_only', -1), ('ism_only', 3)]:
    for seed, scattering in itertools.product([123, 456], [0., 1e-12]):
        pra.random.seed(numpy=seed, libroom=seed)
        room = pra.ShoeBox([6., 6., 2.4], fs=fs, max_order=order,
                           materials=pra.Material(.2, scattering=scattering),
                           ray_tracing=mode != 'ism_only', air_absorption=False)
        if mode != 'ism_only':
            room.set_ray_tracing(n_rays=20000, receiver_radius=.1,
                                 time_thres=.25, hist_bin_size=.0005,
                                 energy_thres=1e-8)
        room.add_source([2.4, 3.6, .96], signal=speech)
        room.add_microphone_array(mics)
        room.compute_rir()
        room.simulate()
        for mic in range(3):
            rir = np.asarray(room.rir[mic][0])
            audio = room.mic_array.signals[mic]
            # A separate convolution verifies that the actual simulator used
            # the measured RIR, with no extra SNR normalization or scaling.
            expected = fftconvolve(speech, rir)
            np.testing.assert_allclose(audio[:len(expected)], expected,
                                       rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(audio[len(expected):], 0., atol=1e-12)
            key = f'{mode}_k{order}_seed{seed}_s{scattering}_m{mic}'
            arrays[key+'_rir'] = rir
            arrays[key+'_audio'] = audio
            records.append(dict(key=key, mode=mode, order=order, seed=seed,
                                scattering=scattering, microphone=mic,
                                rir_length=len(rir), audio_length=len(audio),
                                rir_energy=float(rir@rir),
                                audio_rms=float(np.sqrt(np.mean(audio**2))),
                                rir_sha256=hashlib.sha256(rir.tobytes()).hexdigest(),
                                audio_sha256=hashlib.sha256(audio.tobytes()).hexdigest()))
        print('RENDER', mode, order, seed, scattering, flush=True)
result = dict(source=str(source), imported=pra.__file__, native=pra.libroom.__file__,
              native_sha256=hashlib.sha256(Path(pra.libroom.__file__).read_bytes()).hexdigest(),
              commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
              dirty=subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True),
              python=sys.version, numpy=np.__version__, wav=str(wav),
              wav_sha256=hashlib.sha256(wav.read_bytes()).hexdigest(),
              selected_samples=len(speech), sample_rate=fs, records=records)
a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
np.savez_compressed(a.output.with_suffix('.npz'), **arrays)
