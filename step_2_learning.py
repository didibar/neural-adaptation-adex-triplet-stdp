import numpy as np
from brian2 import *
from pathlib import Path
import pickle
from sklearn.decomposition import PCA


# Learn a Brian2 network and save the final synaptic state.
def learn_and_save_network(file_suffix, n_motif, seed, base_path, directory):

    # Initialize a clean Brian2 simulation scope.
    start_scope()

    np.random.seed(seed)

    # Select the input matrix path according to the feature type.
    if file_suffix != "features_normalized":
        matrix_path = base_path / f"step_1/X_{file_suffix}.npy"
    else:
        matrix_path = base_path / f"step_0/X_{file_suffix}.npy"

    # Load the feature matrix used as the training input.
    feature_matrix = np.load(matrix_path)

    n_neurons = feature_matrix.shape[1]        # Number of neurons in the network.
    n_excitatory = int(3/4 * n_neurons)        # Number of excitatory neurons.
    n_inhibitory = int(1/4 * n_neurons)        # Number of inhibitory neurons.

    # Define the AdEx neuron model parameters.
    c_m = 281 * pF
    g_l = 30 * nS
    tau_m = c_m / g_l
    e_l = -70.6 * mV
    v_t = -50.4 * mV
    delta_t = 2 * mV
    v_cut = v_t + 5 * delta_t

    tau_e = 5 * ms
    tau_i = 10 * ms
    e_e = 0 * mV
    e_i = -80 * mV

    # Define the temporal parameters.
    dt = defaultclock.dt  # Simulation time step (100 us = 0.1 ms).

    # Define the stimulus duration.
    stim_duration = 80 * ms

    # Compute the total simulation duration.
    num_rows = feature_matrix.shape[0]     # Number of input patterns.
    total_time = stim_duration * num_rows  # Total simulation duration.

    # Create the simulation time axis.
    time_points = np.arange(0, total_time, dt)

    # Initialize the input array used during training.
    i_values_training = np.zeros((n_neurons, len(time_points)))

    # Generate the order in which patterns are presented.
    order = []
    for jj in range(n_motif):
        for ii in range(0, num_rows // 2, n_motif):
            idx_first = ii + jj
            idx_second = idx_first + num_rows // 2
            if idx_first < num_rows // 2:
                order.append(idx_first)
            if idx_second < num_rows:
                order.append(idx_second)

    print("Indexes order:", order)


    # Populate the input array with the normalized feature matrix.
    for index, row_index in enumerate(order):
        start_time = index * stim_duration
        start_index = int(start_time / dt)
        stim_duration_int = int(stim_duration / dt)
        i_values_training[:, start_index:start_index + stim_duration_int] = feature_matrix[row_index, :, None]

    # Create a TimedArray to provide external input during the simulation.
    i_recorded = TimedArray(i_values_training.T * nA, dt=dt)


    # Define the neuronal dynamics.
    adex_eqs = '''
        dvm/dt = (g_l*(e_l - vm) + g_l*delta_t*exp((vm - v_t)/delta_t) + i_inj - wad + g_e*(e_e-vm) + g_i*(e_i-vm)) / c_m : volt

        dwad/dt = (a*(vm - e_l) - wad)/tauw : amp

        dg_e/dt = -g_e/tau_e : siemens        # Excitatory synaptic conductance.
        dg_i/dt = -g_i/tau_i : siemens        # Inhibitory synaptic conductance.

        i_inj = i_recorded(t, i) : amp
        tauw : second
        a : siemens
        b : amp
        v_reset : volt
    '''

    # Create excitatory and inhibitory neuron groups.
    ge = NeuronGroup(n_excitatory, adex_eqs, threshold = 'vm > v_cut', reset="vm=v_reset; wad+=b", method='euler')
    gi = NeuronGroup(n_inhibitory, adex_eqs, threshold = 'vm > v_cut', reset="vm=v_reset; wad+=b", method='euler')

    # Initialize neuron state variables.
    ge.vm = e_l
    gi.vm = e_l
    ge.g_e = 0 * nS
    ge.g_i = 0 * nS
    gi.g_e = 0 * nS
    gi.g_i = 0 * nS
    gi.wad = 0 * nA
    ge.wad = 0 * nA

    # Assign regular-spiking parameters to excitatory neurons.
    ge.tauw = 144 * ms
    ge.a = 4 * nS
    ge.b = 0.0805 * nA
    ge.v_reset = -70.6 * mV
    # Assign fast-spiking parameters to inhibitory neurons.
    gi.tauw = 300 * ms
    gi.a = 2 * c_m / (144 * ms)
    gi.b = 0 * nA
    gi.v_reset = -70.6 * mV

    # Define triplet Spike-Timing Dependent Plasticity (t-STDP) parameters.
    tau_pre = 16.8 * ms
    tau_post = 33.7 * ms
    tau_ypre = 144 * ms
    tau_ypost = 1e-6 * ms

    wmax_e = 10   # Maximum synaptic weight for excitatory synapses.
    wmax_i = 15   # Maximum synaptic weight for inhibitory synapses.

    a2_plus = 0
    a2_minus = - 6.5e-3
    a3_plus = 7.1e-3
    a3_minus = - 0

    tstdp_eqs = '''
        w : 1
        dapre/dt = -apre / tau_pre : 1     (event-driven)    # Pre-synaptic trace.
        dapost/dt = -apost / tau_post : 1  (event-driven)    # Post-synaptic trace.
        dypre/dt = -ypre / tau_ypre : 1    (event-driven)    # Slow pre-synaptic trace.
        dypost/dt = -ypost / tau_ypost : 1 (event-driven)    # Slow post-synaptic trace.
        update_weights : 1                                   # Flag used to enable or disable STDP.
    '''
    # Define pre- and post-synaptic updates for excitatory and inhibitory synapses.
    on_pre_eqs_ec ='''
    g_e_post += w * nS                                       # Add excitatory conductance to the post-synaptic neuron.
    apre += a2_plus
    ypre += a3_plus
    w = clip(w + (apost  + ypost)*update_weights, 0, wmax_e)
    '''
    on_post_eqs_ec ='''
    apost += a2_minus
    ypost += a3_minus
    w = clip(w + (apre + ypre)*update_weights, 0, wmax_e)
    '''
    on_pre_eqs_in ='''
    g_i_post += w * nS                                       # Add inhibitory conductance to the post-synaptic neuron.
    apre += a2_plus
    ypre += a3_plus
    w = clip(w + (apost  + ypost)*update_weights, 0, wmax_i)
    '''
    on_post_eqs_in ='''
    apost += a2_minus
    ypost += a3_minus
    w = clip(w + (apre + ypre)*update_weights, 0, wmax_i)
    '''

    # Create synaptic connections.
    c_ee = Synapses(ge, ge, model = tstdp_eqs, on_pre = on_pre_eqs_ec, on_post = on_post_eqs_ec, method='euler')
    c_ee.connect(p=0.065)  # p(EE) = 6.5% - excitatory-to-excitatory connection.
    c_ee.w = 'rand() * 0.25'

    c_ei = Synapses(ge, gi, model = tstdp_eqs, on_pre = on_pre_eqs_ec, on_post = on_post_eqs_ec, method='euler')
    c_ei.connect(p=0.200)  # p(EI) = 20% - excitatory-to-inhibitory connection.
    c_ei.w = 'rand() * 0.25'

    c_ie = Synapses(gi, ge, model = tstdp_eqs, on_pre = on_pre_eqs_in, on_post = on_post_eqs_in, method='euler')
    c_ie.connect(p=0.275)  # p(IE) = 27.5% - inhibitory-to-excitatory connection.
    c_ie.w = 'rand() * 0.25'

    c_ii = Synapses(gi, gi, model = tstdp_eqs, on_pre = on_pre_eqs_in, on_post = on_post_eqs_in, method='euler')
    c_ii.connect(p=0.100)  # p(II) = 10% - inhibitory-to-inhibitory connection.
    c_ii.w = 'rand() * 0.25'

    # Set synaptic delays.
    c_ee.delay = '(1 + 0.5*rand()) * ms'
    c_ei.delay = '(1 + 0.5*rand()) * ms'
    c_ie.delay = '(0.5 + 0.5*rand()) * ms'
    c_ii.delay = '(0.5 + 0.5*rand()) * ms'

    # Enable t-STDP weight updates.
    c_ee.update_weights = 1
    c_ei.update_weights = 1
    c_ie.update_weights = 1
    c_ii.update_weights = 1

    # Run the training simulation.
    run(total_time, report='text')

    # Save the final network state.
    w_ee_last = c_ee.w[:, -1].copy()
    w_ei_last = c_ei.w[:, -1].copy()
    w_ie_last = c_ie.w[:, -1].copy()
    w_ii_last = c_ii.w[:, -1].copy()

    network_state = {
        "C_ee_i": c_ee.i[:],
        "C_ee_j": c_ee.j[:],
        "C_ei_i": c_ei.i[:],
        "C_ei_j": c_ei.j[:],
        "C_ie_i": c_ie.i[:],
        "C_ie_j": c_ie.j[:],
        "C_ii_i": c_ii.i[:],
        "C_ii_j": c_ii.j[:],
        "W_ee": w_ee_last,
        "W_ei": w_ei_last,
        "W_ie": w_ie_last,
        "W_ii": w_ii_last,
    }

    network_state_pt = directory / f"{file_suffix}.pkl"
    with open(network_state_pt, "wb") as f:
        pickle.dump(network_state, f)
