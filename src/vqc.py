import pennylane as qml
import torch
import torch.nn as nn
import numpy as np

# 4 qubit device
dev = qml.device('default.qubit', wires=4)

@qml.qnode(dev, interface='torch')
def vqc_circuit(inputs, weights):
    # inputs: [s0, s1, c0, c1] — sensing(2) + comm(2) values as angles
    # weights: (4, 4) — 4 layers x 4 qubits trainable RY rotations

    # --- Layer 1: Angle Encoding ---
    # sensing branch → qubit 0, 1
    qml.RY(inputs[0], wires=0)
    qml.RY(inputs[1], wires=1)
    # comm branch → qubit 2, 3
    qml.RY(inputs[2], wires=2)
    qml.RY(inputs[3], wires=3)

    # --- Layer 2: Local Entanglement (within branch) ---
    qml.RY(weights[0, 0], wires=0)
    qml.RY(weights[0, 1], wires=1)
    qml.RY(weights[0, 2], wires=2)
    qml.RY(weights[0, 3], wires=3)
    qml.CNOT(wires=[0, 1])   # sensing qubits entangle
    qml.CNOT(wires=[2, 3])   # comm qubits entangle

    # --- Layer 3: Cross Branch Coupling ---
    qml.RY(weights[1, 0], wires=0)
    qml.RY(weights[1, 1], wires=1)
    qml.RY(weights[1, 2], wires=2)
    qml.RY(weights[1, 3], wires=3)
    qml.CNOT(wires=[1, 2])   # sensing → comm
    qml.CNOT(wires=[3, 0])   # comm → sensing

    # --- Layer 4: Global Mixing (ring coupling) ---
    qml.RY(weights[2, 0], wires=0)
    qml.RY(weights[2, 1], wires=1)
    qml.RY(weights[2, 2], wires=2)
    qml.RY(weights[2, 3], wires=3)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 0])   # ring closed

    # --- Final trainable rotation before measurement ---
    qml.RY(weights[3, 0], wires=0)
    qml.RY(weights[3, 1], wires=1)
    qml.RY(weights[3, 2], wires=2)
    qml.RY(weights[3, 3], wires=3)

    # measure all 4 qubits
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]


class DualBranchVQC(nn.Module):
    def __init__(self):
        super().__init__()

        # trainable weights: 4 layers x 4 qubits
        self.weights = nn.Parameter(
            torch.randn(4, 4) * 0.01
        )

        # project sensing output (2,) to 2 angle inputs scaled to [-pi, pi]
        self.sensing_proj = nn.Linear(2, 2)

        # project comm output (1,) to 2 angle inputs
        self.comm_proj = nn.Linear(1, 2)

        # output heads after VQC measurement
        # qubit 0,1 → sensing decision
        self.sensing_out = nn.Linear(2, 2)
        # qubit 2,3 → comm decision
        self.comm_out    = nn.Linear(2, 1)

    def forward(self, sensing, comm):
        # sensing: (batch, 2)
        # comm:    (batch, 1)

        batch_size = sensing.shape[0]
        sensing_angles = torch.tanh(self.sensing_proj(sensing)) * torch.pi  # (batch, 2)
        comm_angles    = torch.tanh(self.comm_proj(comm))       * torch.pi  # (batch, 2)

        outputs = []
        for i in range(batch_size):
            inputs = torch.cat([sensing_angles[i], comm_angles[i]])  # (4,)
            result = vqc_circuit(inputs, self.weights)               # list of 4
            result = torch.stack(result)                             # (4,)
            outputs.append(result)

        outputs = torch.stack(outputs).float()   # (batch, 4) cast to float32   # (batch, 4)

        # split by branch
        sensing_meas = outputs[:, 0:2]  # (batch, 2) — qubits 0,1
        comm_meas    = outputs[:, 2:4]  # (batch, 2) — qubits 2,3

        sensing_final = self.sensing_out(sensing_meas)  # (batch, 2)
        comm_final    = self.comm_out(comm_meas)         # (batch, 1)

        return sensing_final, comm_final