# PR451: the vanishing-scattering limit

Tested base ff7d61f and head c04e851 with separately rebuilt native extensions,
Python3.12.12, NumPy2.5.4, Linux x86_64. Both Python and native import paths are
asserted by the workers. `candidate.patch` applies the maintainer's suggested
unconditional `specular_counter <= ism_order` guard; it is a review experiment,
not a competing pull request or a complete scattering-model validation.

For a 6x6x2.4m shoebox, max_order=1 and the first microphone at [1.6,3,.9],
changing all walls from scattering=0 to 1e-12 changes early ray energy on the
head from 0 to 0.31985920667648315. At max_order2/3 the corresponding jumps are
0.15358397364616394/0.08191566914319992. These bins end before the first possible
uncovered specular arrival. `hybrid_probe.py` enumerates mirrored shoebox
images with integer reflection counts independently of the project's
image-source engine.

The rate is not a physical effect of weak scattering: wall directions use
float32, so a 1e-12 perturbation rounds away, and next-event diffuse estimates
are below the configured energy threshold. But `does_scatter` is true for any
positive coefficient, so the new boolean switches back to the old logging
threshold. The image-source energy at orderK is still essentially unchanged.
Ordinary scattering=0.1 also leaves substantial early energy (0.37726 on head
versus0.04139 with the unconditional guard), though the complete physical
accounting at finite scattering is beyond this continuity test.

`test_scattering_limit.py` adds three material-limit checks and nine controls
which drive the actual native single-ray loop. In the controls, absorption
stops the ray after its first wall, and a scalar solid-angle calculation
predicts the genuine first-wall diffuse microphone estimate. Those nine pass
on the reviewed head and the candidate, showing that the unconditional hit
guard retains the separate `scat_ray` contribution.

Head: three limit failures, nine diffuse controls pass. Candidate: all twelve
pass. The author's six tests all pass on head and fail on base. Candidate
passes the original108 native tests and the author's four no-scattering
tests; its two tests explicitly retaining the old scattered threshold fail.
Those tests would need to change with the proposed guard. A full package
suite, 2D physical-scattering validation and perceptual quality are not claimed.

`render_workload.py` calls `compute_rir` and `simulate` on a one-second,
16kHz excerpt of the shipped CMU ARCTIC aew_a0001 recording. The raw WAV is
only read and is not redistributed. A separate SciPy convolution checks the
actual microphone signals without SNR rescaling. Across orders0..3, two seeds,
three microphones and both material values, plus pure-RT and pure-ISM controls,
each revision produces72 complete RIR/audio records.

On head,18 of36 material pairs differ: relative waveform RMS difference reaches
0.4652543898099719 and total waveform gain differs by up to1.2256051363797504dB.
Base has the same already-duplicated result on both material values. Candidate
restores exact equality for all36 pairs. Its complete RIRs and rendered samples
are identical to head in54/72 records, including every zero-scattering case,
all pure-RT/pure-ISM cases and all order0 cases. This is one recorded excerpt
under controlled synthetic rooms, not a listening study or ASR metric.

## Commands

Build each worktree's native extensions separately before importing it:

```sh
CMAKE_BUILD_PARALLEL_LEVEL=4 python setup.py build_ext --inplace
python review/451/hybrid_probe.py --source /absolute/source --output /tmp/probe.json
python review/451/render_workload.py --source /absolute/source --output /tmp/render.json
PYTHONPATH=/absolute/source python -m pytest review/451/test_scattering_limit.py -q
```

The scripts use explicit source selection. The JSON files record native hashes,
WAV hash and array hashes. Full arrays, build logs, failing assertions and the
control comparison remain in the contribution workspace. No author branch was
changed. A native rebuild is required for the candidate C++ change.
