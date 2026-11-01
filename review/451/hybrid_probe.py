"""Test the zero-scattering limit of the native hybrid ray tracer.

Independent shoebox image enumeration supplies the earliest excluded arrival;
no image-source output is used to choose the histogram cutoff.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
source = a.source.resolve()
sys.path.insert(0, str(source))
import pyroomacoustics as pra

assert Path(pra.__file__).resolve().parent.parent == source
assert Path(pra.libroom.__file__).resolve().parent.parent == source


def images_at_order(dimensions, position, order):
    images = []
    for shifts in itertools.product(range(-order, order + 1), repeat=3):
        for parity in itertools.product([0, 1], repeat=3):
            if sum(abs(2*n-q) for n, q in zip(shifts, parity)) == order:
                images.append(2*np.asarray(dimensions)*shifts
                              + (1-2*np.asarray(parity))*position)
    return np.asarray(images)


dimensions = [6., 6., 2.4]
src = np.array([2.4, 3.6, .96])
mics = np.array([[1.6, 3., .9], [3., 1.6, 1.8], [4.4, 3., .9]]).T
records = []
arrays = {}
for order, scattering in itertools.product([0, 1, 2, 3], [0., 1e-12, 1e-8, .1, 1.]):
    pra.random.seed(numpy=123, libroom=123)
    room = pra.ShoeBox(dimensions, fs=16000, max_order=order,
                       materials=pra.Material(.2, scattering=scattering),
                       ray_tracing=True, air_absorption=False)
    room.set_ray_tracing(n_rays=20000, receiver_radius=.1, time_thres=.25,
                         hist_bin_size=.0005, energy_thres=1e-8)
    room.add_source(src)
    room.add_microphone_array(mics)
    room.ray_tracing()
    points = images_at_order(dimensions, src, order+1)
    row = dict(order=order, scattering=scattering, microphones=[])
    for mic_index in range(mics.shape[1]):
        d_min = np.linalg.norm(points-mics[:, mic_index], axis=1).min()
        radius = room.rt_args['receiver_radius']
        bin_dist = room.rt_args['hist_bin_size']*room.c
        cutoff = int((np.sqrt(d_min**2-radius**2)-.01)/bin_dist)
        hist = np.asarray(room.rt_histograms[mic_index][0][0])
        key = f'k{order}_s{scattering}_m{mic_index}'
        arrays[key] = hist
        record = dict(cutoff=cutoff, first_uncovered_distance=float(d_min),
                      early_energy=float(hist[..., :cutoff].sum()),
                      late_energy=float(hist[..., cutoff:].sum()),
                      total_energy=float(hist.sum()),
                      histogram_hash=hashlib.sha256(hist.tobytes()).hexdigest())
        row['microphones'].append(record)
    records.append(row)
    print('CASE', order, scattering,
          [r['early_energy'] for r in row['microphones']], flush=True)

result = dict(source=str(source), imported=pra.__file__, native=pra.libroom.__file__,
              native_sha256=hashlib.sha256(Path(pra.libroom.__file__).read_bytes()).hexdigest(),
              python=sys.version, numpy=np.__version__,
              commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
              dirty=subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True),
              records=records)
a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
np.savez_compressed(a.output.with_suffix('.npz'), **arrays)
