Quantum Algorithms

This repository contains implementations of foundational quantum algorithms using Qiskit.


Grover's Algorithm:

Initialize qubits in uniform superposition using Hadamard gates.

Apply the oracle that flips the phase of the marked state(s).

Apply the diffusion operator (inversion about the mean).

Repeat steps 2–3 approximately √N times.

Measure the qubits to obtain the marked element with high probability.



Quantum Fourier Transform (QFT):

Apply a Hadamard gate to the first qubit.

Apply controlled-phase rotations between the first qubit and the rest.

Repeat recursively for remaining qubits.

Swap qubits to reverse their order.

The result is the quantum Fourier transform of the input state.



Bernstein-Vazirani Algorithm:

Initialize all qubits to |0⟩ and one ancilla to |1⟩.

Apply Hadamard gates to all qubits.

Use the oracle to compute the hidden bitstring s via XOR (phase kickback).

Apply Hadamard gates again to the input qubits.

Measure — the output reveals the hidden string s in one query.
