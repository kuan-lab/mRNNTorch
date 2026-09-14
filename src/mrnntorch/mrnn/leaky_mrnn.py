"""mRNN core module.

Implements the multi-regional recurrent neural network (mRNN) building blocks
and step-wise dynamics, along with helpers for connectivity, constraints, and
initialization."""

import torch
from typing import Tuple
from mrnntorch.mrnn.mrnn_base import mRNNBase


DEFAULTS_MRNN = {
    "config": None,
    "activation": "relu",
    "noise_level_act": 0.01,
    "noise_level_inp": 0.01,
    "rec_constrained": True,
    "inp_constrained": True,
    "batch_first": True,
    "spectral_radius": None,
    "config_finalize": True,
    "device": "cuda",
    "dt": 10,
    "tau": 100,
    "tau_inh": None,
}


def linear(x):
    """Return ``x`` unchanged."""
    return x


class mRNN(mRNNBase):
    """Leaky multi-regional RNN with separate pre-activation and activation states."""

    def __init__(
        self,
        config: str = DEFAULTS_MRNN["config"],
        activation: str = DEFAULTS_MRNN["activation"],
        noise_level_act: float = DEFAULTS_MRNN["noise_level_act"],
        noise_level_inp: float = DEFAULTS_MRNN["noise_level_inp"],
        rec_constrained: bool = DEFAULTS_MRNN["rec_constrained"],
        inp_constrained: bool = DEFAULTS_MRNN["inp_constrained"],
        batch_first: bool = DEFAULTS_MRNN["batch_first"],
        spectral_radius: float = DEFAULTS_MRNN["spectral_radius"],
        config_finalize: bool = DEFAULTS_MRNN["config_finalize"],
        device: str = DEFAULTS_MRNN["device"],
        dt: float = DEFAULTS_MRNN["dt"],
        tau: float = DEFAULTS_MRNN["tau"],
        tau_inh: float | None = DEFAULTS_MRNN["tau_inh"],
    ):
        """Initialize a leaky multi-regional RNN.

        Args:
            config (str | None): Optional JSON config path.
            activation (str): Hidden activation function name.
            noise_level_act (float): Hidden-state noise scale.
            noise_level_inp (float): Input noise scale.
            rec_constrained (bool): Whether recurrent weights obey Dale's law.
            inp_constrained (bool): Whether input weights obey Dale's law.
            batch_first (bool): Whether sequences are batch-major.
            spectral_radius (float | None): Optional recurrent spectral-radius target.
            config_finalize (bool): Whether to finalize connectivity after config load.
            device (str): Torch device string.
            dt (float): Discretization step.
            tau (float): Time constant for excitatory (``sign == "pos"``) units.
            tau_inh (float | None): Separate time constant for inhibitory
                (``sign == "neg"``) units. If ``None`` (default), every unit
                shares ``tau`` and ``self.alpha`` is the original scalar
                ``dt / tau`` — fully backward compatible. If set, ``self.alpha``
                becomes a per-unit vector with ``dt / tau_inh`` on inhibitory
                units (``tau_inh < tau`` gives *faster* inhibition, recovering
                the adiabatic-inhibition regime of the reduced theory).
        """
        super(mRNN, self).__init__(
            config,
            activation,
            noise_level_act,
            noise_level_inp,
            rec_constrained,
            inp_constrained,
            batch_first,
            spectral_radius,
            config_finalize,
            device,
        )
        self.dt = dt
        self.tau = tau
        self.tau_inh = tau_inh
        # Scalar reference gain (excitatory dt/tau). Noise constants always use
        # this scalar so that (a) per-unit alpha never breaks input-noise
        # dimensions and (b) noise statistics are unchanged from the scalar-tau
        # baseline. See mrnn_base.{hid,inp}_noise_const.
        self.alpha_scalar = dt / tau
        # Default: original scalar behaviour. Only becomes a per-unit vector if
        # a separate inhibitory tau is requested AND regions already exist
        # (config-driven build). Manual builders can call set_inhibitory_tau()
        # after finalize_connectivity().
        self.alpha = dt / tau
        if tau_inh is not None and len(self.region_dict) > 0:
            self.set_inhibitory_tau(tau_inh)

    def set_inhibitory_tau(self, tau_inh: float) -> torch.Tensor:
        """Give inhibitory-region units a separate time constant ``tau_inh``.

        Replaces the scalar ``self.alpha`` with a per-unit vector of shape
        ``(total_num_units,)``: excitatory (``sign == "pos"``) units keep
        ``dt / tau``; inhibitory (``sign == "neg"``) units use ``dt / tau_inh``.
        The vector broadcasts over the batch dimension in :meth:`forward`, so no
        other code path changes. Call after all recurrent regions are added
        (automatically invoked from ``__init__`` on the config-driven path).
        """
        self.tau_inh = tau_inh
        self.alpha = self._build_alpha_vector(self.dt, self.tau, tau_inh)
        return self.alpha

    def _build_alpha_vector(self, dt, tau, tau_inh) -> torch.Tensor:
        """Per-unit update gain aligned with the hidden-state unit ordering."""
        assert self.total_num_units > 0 and len(self.region_dict) > 0, (
            "set_inhibitory_tau() requires recurrent regions to be defined first"
        )
        alpha = torch.full((self.total_num_units,), dt / tau, device=self.device)
        idx = 0
        for region in self.region_dict.values():
            n = region.num_units
            if region.sign == "neg":          # inhibitory outputs
                alpha[idx:idx + n] = dt / tau_inh
            idx += n
        return alpha

    def batched_initial_condition(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return batched initial pre-activation and activation states."""
        xn = self.initial_condition.unsqueeze(0).repeat(batch_size, 1)
        hn = self.initial_condition.unsqueeze(0).repeat(batch_size, 1)
        return xn, hn

    def forward(
        self,
        inp: torch.Tensor,
        x0: torch.Tensor,
        h0: torch.Tensor | None = None,
        stim_input: torch.Tensor | None = None,
        noise: bool = False,
        W_rec: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the recurrent dynamics over a sequence.

        Discretized update: ``x_{t+1} = x_t + alpha * (-x_t + W_rec h_t + W_inp u_t + b + noise)``
        and ``h_{t+1} = activation(x_{t+1})``.

        Args:
            inp (torch.Tensor): Input sequence. Shape ``[B, T, I]`` if batch_first
                else ``[T, B, I]``.
            x0 (torch.Tensor): Initial pre-activation hidden state, shape ``[B, H]``.
            h0 (torch.Tensor): Initial activation, shape ``[B, H]``.
            *args (torch.Tensor): Optional additive inputs with same temporal layout
                as ``inp`` and feature size ``H``.
            noise (bool): If True, add Gaussian noise to hidden state and inputs.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(x_seq, h_seq)`` sequences matching
            the temporal layout of ``inp``.
        """
        assert len(self.region_dict) > 0
        assert len(self.inp_dict) > 0
        assert self.rec_finalized or self.inp_finalized, (
            "Recurrent or input weights are not finalized, \
            call finalize_connectivity() in your custom model definition"
        )

        if inp.dim() != 3:
            raise Exception(
                "input must be 3 dimensional, \
                            [batch, time, units] for batch_first=True, \
                            and [time, batch, units] otherwise]."
            )
        if x0.dim() != 2:
            raise Exception("x0 must be 2 dimensional, [batch, units].")

        if stim_input is not None:
            if stim_input.dim() != 3:
                raise Exception(
                    "stim_input must be 3 dimensional, \
                                [batch, time, units] for batch_first=True, \
                                and [time, batch, units] otherwise]."
                )

        if W_rec is None:
            # Apply Dale's Law if constrained
            if self.rec_constrained:
                W_rec = self.apply_dales_law(
                    self.W_rec, self.W_rec_mask, self.W_rec_sign_matrix
                )
            else:
                W_rec = self.W_rec * self.W_rec_mask
        assert isinstance(W_rec, torch.Tensor)

        # Apply to input weights as well
        if self.inp_constrained:
            W_inp = self.apply_dales_law(
                self.W_inp, self.W_inp_mask, self.W_inp_sign_matrix
            )
        else:
            W_inp = self.W_inp * self.W_inp_mask

        baseline_inp = self.tonic_inp

        xn_next = x0
        hn_next = self.activation(x0) if h0 is None else h0
        assert isinstance(hn_next, torch.Tensor)

        if self.batch_first:
            # If batch first then batch is first dim
            batch_shape = inp.shape[0]
            seq_len = inp.shape[1]
            shape = (batch_shape, seq_len, self.total_num_units)
        else:
            # If not batch first then seq_len is first dim
            seq_len = inp.shape[0]
            batch_shape = inp.shape[1]
            shape = (seq_len, batch_shape, self.total_num_units)

        # Create lists for xs and hns
        new_hs = torch.empty(size=shape, device=self.device)
        new_xs = torch.empty(size=shape, device=self.device)

        # Process sequence
        for t in range(seq_len):
            # Gather input at current timestep
            if self.batch_first:
                inp_t = inp[:, t, :]
            else:
                inp_t = inp[t, :, :]

            # Sample from normal distribution and scale by constant term
            if noise:
                # Separate noise levels will be applied to each neuron/input
                hid_noise = self._hid_noise(batch_shape)
                inp_noise = self._inp_noise(batch_shape)
            else:
                hid_noise = inp_noise = 0

            """
            Update hidden state
            Discretized equation of the form: 
                x_(t+1) = x_t + alpha * (-x_t + Wh + W_ix + b)
            """

            xn_next = xn_next + self.alpha * (
                -xn_next
                + (W_rec @ hn_next.T).T
                + (W_inp @ (inp_t + inp_noise).T).T
                + baseline_inp
                + hid_noise
            )

            if stim_input is not None:
                # Add any additional arg inputs (stim inputs typically)
                if self.batch_first:
                    xn_next = xn_next + self.alpha * stim_input[:, t, :]
                else:
                    xn_next = xn_next + self.alpha * stim_input[t, :, :]

            """
            Compute activation
            Activation of the form: 
                h_t = sigma(x_t)
            """

            hn_next = self.activation(xn_next)

            if self.batch_first:
                new_xs[:, t, :] = xn_next
                new_hs[:, t, :] = hn_next
            else:
                new_xs[t, :, :] = xn_next
                new_hs[t, :, :] = hn_next

        return new_xs, new_hs
