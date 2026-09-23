import math
import pennylane as qml
import torch
import torch.nn as nn


class QuantumMetricMixin:
    """Shared pure-state metric implementation for the actor and critics."""

    def _build_state_circuit(self):
        @qml.qnode(self.quantum_device, interface="torch", diff_method="backprop")
        def circuit(encoded_angles, var_params):
            for layer in range(self.reupload_layers):
                for qubit in range(self.num_qubits):
                    qml.RY(encoded_angles[layer, qubit, 0], wires=qubit)
                    qml.RZ(encoded_angles[layer, qubit, 1], wires=qubit)
                for qubit in range(self.num_qubits):
                    qml.Rot(
                        var_params[layer, qubit, 0],
                        var_params[layer, qubit, 1],
                        var_params[layer, qubit, 2],
                        wires=qubit,
                    )
                for qubit in range(self.num_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
                qml.CNOT(wires=[self.num_qubits - 1, 0])

            final_layer = var_params[self.reupload_layers]
            for qubit in range(self.num_qubits):
                qml.Rot(
                    final_layer[qubit, 0],
                    final_layer[qubit, 1],
                    final_layer[qubit, 2],
                    wires=qubit,
                )
            return qml.state()

        return circuit

    def _single_quantum_fisher(self, encoded_angles, flat_params):
        """Calculate the pure-state Fubini-Study metric for one sample."""
        encoded_angles = encoded_angles.detach()
        parameter_shape = self.var_params.shape
        parameter_count = flat_params.numel()

        def real_imag_state(parameters):
            shaped = parameters.reshape(parameter_shape)
            state = self.metric_circuit(encoded_angles, shaped).reshape(-1)
            return torch.cat((state.real, state.imag), dim=0)

        state_parts = real_imag_state(flat_params)
        state_count = state_parts.numel() // 2
        jacobian = torch.autograd.functional.jacobian(
            real_imag_state,
            flat_params,
            create_graph=False,
            vectorize=False,
        ).reshape(2 * state_count, parameter_count)

        real_dtype = state_parts.dtype
        state = torch.complex(
            state_parts[:state_count], state_parts[state_count:]
        )
        state_jacobian = torch.complex(
            jacobian[:state_count].to(dtype=real_dtype),
            jacobian[state_count:].to(dtype=real_dtype),
        )
        overlap = state_jacobian.conj().transpose(0, 1) @ state
        metric = (
            state_jacobian.conj().transpose(0, 1) @ state_jacobian
            - torch.outer(overlap, overlap.conj())
        ).real
        return 0.5 * (metric + metric.transpose(0, 1))

    def quantum_fisher(self, encoded_angles, max_samples=None):
        """Return a batch-averaged pure-state metric for ``var_params``."""
        if encoded_angles.ndim == 3:
            encoded_angles = encoded_angles.unsqueeze(0)
        sample_count = encoded_angles.shape[0]
        if max_samples is not None:
            sample_count = min(sample_count, max(int(max_samples), 1))
        if sample_count < 1:
            raise ValueError("encoded_angles must contain at least one sample")

        flat_params = self.var_params.detach().reshape(-1).clone()
        flat_params.requires_grad_(True)
        metric = None
        for sample in encoded_angles[:sample_count]:
            sample_metric = self._single_quantum_fisher(sample, flat_params)
            metric = sample_metric if metric is None else metric + sample_metric
        metric = metric / float(sample_count)
        return metric.detach().to(device=self.var_params.device, dtype=torch.float64)


class QuantumVariationalPolicy(QuantumMetricMixin, nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dim,
        log_std_min,
        log_std_max,
        log_std_init,
        num_qubits=6,
        reupload_layers=2,
    ):
        super().__init__()
        if reupload_layers < 1:
            raise ValueError("reupload_layers must be >= 1")
        encoding_slots = reupload_layers * num_qubits * 2
        if encoding_slots < obs_dim:
            raise ValueError(
                "Angle encoding capacity is too small: "
                f"{reupload_layers} reupload layers * {num_qubits} qubits * 2 angles "
                f"= {encoding_slots} slots, but obs_dim={obs_dim}. "
                "Increase num_qubits or reupload_layers."
            )
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.num_qubits = num_qubits
        self.reupload_layers = reupload_layers
        self.num_layers = reupload_layers + 1
        self.encoding_slots = encoding_slots

        self.input_norm = nn.LayerNorm(obs_dim)
        self.angle_scales = nn.Parameter(torch.ones(obs_dim))
        self.angle_bias = nn.Parameter(torch.zeros(obs_dim))

        self.var_params = nn.Parameter(0.02 * torch.randn(self.num_layers, self.num_qubits, 3))

        self.measure_dim = self.num_qubits * 2
        self.mean_readout = nn.Linear(self.measure_dim, action_dim)
        self.log_std_readout = nn.Linear(self.measure_dim, action_dim)

        nn.init.uniform_(self.log_std_readout.weight, -1e-3, 1e-3)
        init_bias = float(max(min(log_std_init, self.log_std_max), self.log_std_min))
        nn.init.constant_(self.log_std_readout.bias, init_bias)

        self.quantum_device = qml.device("default.qubit", wires=self.num_qubits)
        self.quantum_circuit = self._build_quantum_circuit()
        self.metric_circuit = self._build_state_circuit()

    def _build_quantum_circuit(self):
        @qml.qnode(self.quantum_device, interface="torch", diff_method="backprop")
        def circuit(encoded_angles, var_params):
            for layer in range(self.reupload_layers):
                for qubit in range(self.num_qubits):
                    qml.RY(encoded_angles[..., layer, qubit, 0], wires=qubit)
                    qml.RZ(encoded_angles[..., layer, qubit, 1], wires=qubit)

                for qubit in range(self.num_qubits):
                    qml.Rot(
                        var_params[layer, qubit, 0],
                        var_params[layer, qubit, 1],
                        var_params[layer, qubit, 2],
                        wires=qubit,
                    )

                for qubit in range(self.num_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
                qml.CNOT(wires=[self.num_qubits - 1, 0])

            final_layer = var_params[self.reupload_layers]
            for qubit in range(self.num_qubits):
                qml.Rot(
                    final_layer[qubit, 0],
                    final_layer[qubit, 1],
                    final_layer[qubit, 2],
                    wires=qubit,
                )
            z_measurements = [qml.expval(qml.PauliZ(qubit)) for qubit in range(self.num_qubits)]
            x_measurements = [qml.expval(qml.PauliX(qubit)) for qubit in range(self.num_qubits)]
            return z_measurements + x_measurements

        return circuit

    def _encode_obs(self, obs):
        obs = self.input_norm(obs)
        feature_angles = math.pi * torch.tanh(obs * self.angle_scales + self.angle_bias)
        if self.encoding_slots > self.obs_dim:
            padding = feature_angles.new_zeros(obs.shape[0], self.encoding_slots - self.obs_dim)
            feature_angles = torch.cat((feature_angles, padding), dim=-1)
        return feature_angles.view(obs.shape[0], self.reupload_layers, self.num_qubits, 2)
  

    def _run_quantum_circuit(self, encoded_angles):
        measured = self.quantum_circuit(encoded_angles, self.var_params)
        if isinstance(measured, (tuple, list)):
            measured = torch.stack(list(measured), dim=-1)
        if measured.ndim == 1:
            measured = measured.unsqueeze(0)
        return measured.to(dtype=encoded_angles.dtype)

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs = obs.float().view(batch_size, -1)
        encoded_angles = self._encode_obs(obs)
        measured = self._run_quantum_circuit(encoded_angles)
        mean = self.mean_readout(measured)
        log_std = self.log_std_readout(measured)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs, deterministic=False):
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std)

        if deterministic:
            pre_tanh = mean
            action = torch.tanh(pre_tanh)
            return action, None

        normal = torch.distributions.Normal(mean, std)
        pre_tanh = normal.rsample()
        action = torch.tanh(pre_tanh)

        log_prob = normal.log_prob(pre_tanh)
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob
    

class QuantumQNetwork(QuantumMetricMixin, nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim,
        num_qubits=8,
        reupload_layers=2,
    ):
        super().__init__()
        if reupload_layers < 1:
            raise ValueError("reupload_layers must be >= 1")

        input_dim = state_dim + action_dim
        encoding_slots = reupload_layers * num_qubits * 2
        if encoding_slots < input_dim:
            raise ValueError(
                "Critic angle encoding capacity is too small: "
                f"{reupload_layers} reupload layers * {num_qubits} qubits * 2 angles "
                f"= {encoding_slots} slots, but critic input_dim={input_dim}. "
                "Increase quantum_critic_num_qubits or quantum_critic_reupload_layers."
            )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_qubits = num_qubits
        self.reupload_layers = reupload_layers
        self.num_layers = reupload_layers + 1
        self.encoding_slots = encoding_slots
        self.measure_dim = self.num_qubits * 2

        self.angle_scales = nn.Parameter(torch.ones(input_dim))
        self.angle_bias = nn.Parameter(torch.zeros(input_dim))
        self.action_angle_scales = nn.Parameter(torch.zeros(self.encoding_slots, action_dim))
        self.var_params = nn.Parameter(0.02 * torch.randn(self.num_layers, self.num_qubits, 3))
        self._init_action_angle_scales()


        self.value_readout = nn.Linear(self.measure_dim, 1)
        nn.init.uniform_(self.value_readout.weight, -1e-3, 1e-3)
        nn.init.constant_(self.value_readout.bias, 0.0)

        self.quantum_device = qml.device("default.qubit", wires=self.num_qubits)
        self.quantum_circuit = self._build_quantum_circuit()
        self.metric_circuit = self._build_state_circuit()

    def _init_action_angle_scales(self):
        with torch.no_grad():
            self.action_angle_scales.zero_()
            for slot in range(self.encoding_slots):
                self.action_angle_scales[slot, slot % self.action_dim] = 0.25

    def _build_quantum_circuit(self):
        @qml.qnode(self.quantum_device, interface="torch", diff_method="backprop")
        def circuit(encoded_angles, var_params):
            for layer in range(self.reupload_layers):
                for qubit in range(self.num_qubits):
                    qml.RY(encoded_angles[..., layer, qubit, 0], wires=qubit)
                    qml.RZ(encoded_angles[..., layer, qubit, 1], wires=qubit)

                for qubit in range(self.num_qubits):
                    qml.Rot(
                        var_params[layer, qubit, 0],
                        var_params[layer, qubit, 1],
                        var_params[layer, qubit, 2],
                        wires=qubit,
                    )

                for qubit in range(self.num_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
                qml.CNOT(wires=[self.num_qubits - 1, 0])

            final_layer = var_params[self.reupload_layers]
            for qubit in range(self.num_qubits):
                qml.Rot(
                    final_layer[qubit, 0],
                    final_layer[qubit, 1],
                    final_layer[qubit, 2],
                    wires=qubit,
                )
            z_measurements = [qml.expval(qml.PauliZ(qubit)) for qubit in range(self.num_qubits)]
            x_measurements = [qml.expval(qml.PauliX(qubit)) for qubit in range(self.num_qubits)]
            return z_measurements + x_measurements

        return circuit

    def _encode_input(self, x, action):
        feature_preactivations = x * self.angle_scales + self.angle_bias
        if self.encoding_slots > self.input_dim:
            padding = feature_preactivations.new_zeros(
                x.shape[0], self.encoding_slots - self.input_dim
            )
            feature_preactivations = torch.cat((feature_preactivations, padding), dim=-1)
        action_preactivations = action @ self.action_angle_scales.t()
        feature_angles = math.pi * torch.tanh(feature_preactivations + action_preactivations)
        return feature_angles.view(x.shape[0], self.reupload_layers, self.num_qubits, 2)

    def _run_quantum_circuit(self, encoded_angles):
        measured = self.quantum_circuit(encoded_angles, self.var_params)
        if isinstance(measured, (tuple, list)):
            measured = torch.stack(list(measured), dim=-1)
        if measured.ndim == 1:
            measured = measured.unsqueeze(0)
        return measured.to(dtype=encoded_angles.dtype)

    def forward(self, state, action):
        batch_size = state.shape[0]
        state = state.float().view(batch_size, -1)
        action = action.float().view(batch_size, -1)
        x = torch.cat([state, action], dim=-1)
        encoded_angles = self._encode_input(x, action)
        measured = self._run_quantum_circuit(encoded_angles)
        return self.value_readout(measured)
