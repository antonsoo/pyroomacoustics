"""Independent regression for PR451's nonzero-scattering threshold."""

import numpy as np
import pytest

import pyroomacoustics as pra


@pytest.mark.parametrize('order', [1, 2, 3])
def test_vanishing_scattering_retains_specular_limit(order):
    """An imperceptible material change must not restore a full reflection."""
    histograms = []
    for scattering in [0., 1e-12]:
        pra.random.seed(numpy=123, libroom=123)
        room = pra.ShoeBox([6., 6., 2.4], fs=16000, max_order=order,
                           materials=pra.Material(.2, scattering=scattering),
                           ray_tracing=True, air_absorption=False)
        room.set_ray_tracing(n_rays=20000, receiver_radius=.1,
                             time_thres=.25, hist_bin_size=.0005,
                             energy_thres=1e-8)
        room.add_source([2.4, 3.6, .96])
        room.add_microphone([1.6, 3., .9])
        room.ray_tracing()
        histograms.append(np.asarray(room.rt_histograms[0][0][0]))
    # Native arithmetic is float32. At 1e-12 the direction perturbation rounds
    # away, and diffuse estimates lie below the configured energy threshold.
    np.testing.assert_allclose(histograms[0], histograms[1], rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize('order', [1, 2, 3])
@pytest.mark.parametrize('scattering', [.1, .5, 1.])
def test_actual_first_wall_diffuse_estimate_is_retained(order, scattering):
    """The next-event wall estimate is independent of hybrid hit suppression."""
    room = pra.ShoeBox([6., 6., 2.4], fs=16000, max_order=order,
                       materials=pra.Material(.9, scattering=scattering),
                       ray_tracing=True, air_absorption=False)
    room.set_ray_tracing(receiver_radius=.5, time_thres=.1,
                         hist_bin_size=.0005, energy_thres=.2)
    room.add_source([3., 3., 1.2])
    room.add_microphone([2., 4., 1.2])
    # After the first wall the remaining energy is 1000, below the stopping
    # threshold 2000. Its diffuse microphone estimate is nevertheless >0.2,
    # so exactly one next-event estimate reaches the histogram.
    room.room_engine.simul_ray(np.array([1., 0., 0.]),
                                np.array([3., 3., 1.2]), np.array([10000.]))
    hist = room.room_engine.microphones[0].histograms[0].get_hist()
    radius2 = .5**2
    hop = np.sqrt(17.)
    total = 3.+hop
    # Direct scalar solid-angle geometry for the east wall and receiver.
    p_hop = 1.-np.sqrt(1.-radius2/17.)
    p_total = 1.-np.sqrt(1.-radius2/total**2)
    expected = (1000.*scattering*p_hop*(8./hop))/(total**2*p_total)
    np.testing.assert_allclose(np.sum(hist), expected, rtol=3e-5)
