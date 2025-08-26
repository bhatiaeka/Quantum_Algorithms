"""
Quantum Fourier Transform (QFT) Implementation using Qiskit

The QFT is a quantum analog of the discrete Fourier transform, and is a key
component in many quantum algorithms including Shor's algorithm and quantum
phase estimation.
"""

# Install required packages:
# pip install qiskit qiskit-aer matplotlib numpy

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram, circuit_drawer, plot_bloch_multivector
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import cmath


class QuantumFourierTransform:
    """Quantum Fourier Transform implementation and demonstrations."""

    def __init__(self, n_qubits: int):
        """
        Initialize QFT for n qubits.

        Args:
            n_qubits: Number of qubits
        """
        self.n_qubits = n_qubits
        self.N = 2**n_qubits

    def create_qft_circuit(
        self, n_qubits: int = None, with_swaps: bool = True
    ) -> QuantumCircuit:
        """
        Create QFT circuit for n qubits.

        QFT transforms |j⟩ → (1/√N) Σ exp(2πijk/N) |k⟩

        Args:
            n_qubits: Number of qubits (uses self.n_qubits if None)
            with_swaps: Whether to include swap gates at the end

        Returns:
            QFT circuit
        """
        if n_qubits is None:
            n_qubits = self.n_qubits

        qft = QuantumCircuit(n_qubits, name="QFT")

        # Apply QFT
        for j in range(n_qubits):
            # Apply Hadamard to qubit j
            qft.h(j)

            # Apply controlled phase rotations
            for k in range(j + 1, n_qubits):
                angle = np.pi / (2 ** (k - j))
                qft.cp(angle, k, j)

        # Swap qubits to get correct output order
        if with_swaps:
            for j in range(n_qubits // 2):
                qft.swap(j, n_qubits - j - 1)

        return qft

    def create_inverse_qft_circuit(self, n_qubits: int = None) -> QuantumCircuit:
        """
        Create inverse QFT (QFT†) circuit.

        Args:
            n_qubits: Number of qubits

        Returns:
            Inverse QFT circuit
        """
        if n_qubits is None:
            n_qubits = self.n_qubits

        iqft = QuantumCircuit(n_qubits, name="QFT†")

        # Swap qubits first (reverse of QFT)
        for j in range(n_qubits // 2):
            iqft.swap(j, n_qubits - j - 1)

        # Apply inverse QFT operations in reverse order
        for j in reversed(range(n_qubits)):
            # Apply controlled phase rotations (with negative angles)
            for k in reversed(range(j + 1, n_qubits)):
                angle = -np.pi / (2 ** (k - j))
                iqft.cp(angle, k, j)

            # Apply Hadamard
            iqft.h(j)

        return iqft

    def demonstrate_qft_on_basis_state(self, input_state: int) -> Dict:
        """
        Demonstrate QFT on a computational basis state.

        Args:
            input_state: Integer representing basis state (e.g., 3 for |011⟩)

        Returns:
            Dictionary with circuit and results
        """
        # Create circuit
        qreg = QuantumRegister(self.n_qubits, "q")
        creg = ClassicalRegister(self.n_qubits, "c")
        circuit = QuantumCircuit(qreg, creg)

        # Prepare input state
        binary = format(input_state, f"0{self.n_qubits}b")
        for i, bit in enumerate(reversed(binary)):
            if bit == "1":
                circuit.x(i)

        circuit.barrier()

        # Apply QFT
        qft = self.create_qft_circuit()
        circuit.append(qft, range(self.n_qubits))

        # Get statevector before measurement
        simulator = AerSimulator(method="statevector")
        circuit_sv = circuit.copy()
        statevector = Statevector(circuit_sv)
        statevector = statevector.evolve(circuit_sv)

        # Add measurements
        circuit.barrier()
        circuit.measure(qreg, creg)

        return {
            "circuit": circuit,
            "input_state": input_state,
            "binary": binary,
            "statevector": statevector,
            "amplitudes": statevector.data,
        }

    def verify_qft_unitarity(self) -> bool:
        """
        Verify that QFT is unitary (QFT† × QFT = I).

        Returns:
            True if QFT is unitary
        """
        # Create QFT and inverse QFT
        qft = self.create_qft_circuit()
        iqft = self.create_inverse_qft_circuit()

        # Create circuit that applies QFT then QFT†
        circuit = QuantumCircuit(self.n_qubits)
        circuit.append(qft, range(self.n_qubits))
        circuit.append(iqft, range(self.n_qubits))

        # Check if this equals identity
        # Get the unitary matrix
        backend = AerSimulator(method="unitary")
        circuit_compiled = transpile(circuit, backend)
        job = backend.run(circuit_compiled)
        result = job.result()
        unitary = result.get_unitary(circuit_compiled)

        # Check if it's identity (within numerical precision)
        identity = np.eye(2**self.n_qubits)
        is_unitary = np.allclose(unitary, identity)

        return is_unitary

    def classical_dft(self, input_vector: List[complex]) -> List[complex]:
        """
        Classical Discrete Fourier Transform for comparison.

        Args:
            input_vector: Input vector

        Returns:
            DFT of input vector
        """
        N = len(input_vector)
        output = []

        for k in range(N):
            sum_val = 0
            for j in range(N):
                angle = -2 * np.pi * j * k / N
                sum_val += input_vector[j] * cmath.exp(1j * angle)
            output.append(sum_val / np.sqrt(N))

        return output


def demonstrate_qft_basics():
    """Demonstrate basic QFT properties."""

    print("=" * 70)
    print("QUANTUM FOURIER TRANSFORM (QFT) DEMONSTRATION")
    print("=" * 70)

    # Example 1: 2-qubit QFT
    print("\n--- Example 1: 2-qubit QFT Circuit ---")
    qft = QuantumFourierTransform(2)
    circuit = qft.create_qft_circuit()
    print(circuit.draw(output="text"))

    print("\nThe QFT circuit consists of:")
    print("1. Hadamard gates to create superposition")
    print("2. Controlled phase rotations for quantum correlations")
    print("3. Swap gates to correct output ordering")

    # Example 2: QFT on basis states
    print("\n--- Example 2: QFT on Computational Basis States ---")
    qft = QuantumFourierTransform(3)

    for state in [0, 1, 4, 7]:
        result = qft.demonstrate_qft_on_basis_state(state)
        print(f"\nInput: |{result['binary']}⟩ (decimal: {state})")
        print("Output amplitudes (first 4):")

        for i in range(min(4, len(result["amplitudes"]))):
            amp = result["amplitudes"][i]
            magnitude = abs(amp)
            phase = cmath.phase(amp)
            print(
                f"  |{format(i, '03b')}⟩: magnitude={magnitude:.3f}, phase={phase / np.pi:.2f}π"
            )

    # Example 3: Unitarity check
    print("\n--- Example 3: Verifying QFT Properties ---")
    for n in [2, 3, 4]:
        qft = QuantumFourierTransform(n)
        is_unitary = qft.verify_qft_unitarity()
        print(f"{n}-qubit QFT: Unitary? {is_unitary} ✓")

    print("\nQFT† × QFT = I (Identity)")
    print("This confirms QFT is reversible and preserves quantum information")


def demonstrate_qft_patterns():
    """Demonstrate interesting QFT patterns and periodicity."""

    print("\n" + "=" * 70)
    print("QFT PATTERNS AND PERIODICITY")
    print("=" * 70)

    print("\nQFT reveals periodicity in quantum states:")

    # Create periodic state
    print("\n--- Creating a Periodic State ---")
    n_qubits = 4
    circuit = QuantumCircuit(n_qubits)

    # Create superposition of |0⟩ and |8⟩ (period 8)
    circuit.h(0)
    circuit.x(3)
    circuit.cx(0, 3)

    print("Initial state: (|0000⟩ + |1000⟩)/√2")
    print("This has period 8 in the computational basis")

    # Apply QFT
    qft = QuantumFourierTransform(n_qubits)
    qft_circuit = qft.create_qft_circuit()
    circuit.append(qft_circuit, range(n_qubits))

    # Get statevector
    statevector = Statevector(circuit)
    amplitudes = statevector.data

    print("\nAfter QFT, non-zero amplitudes at:")
    for i, amp in enumerate(amplitudes):
        if abs(amp) > 0.01:
            binary = format(i, f"0{n_qubits}b")
            print(f"  |{binary}⟩: {abs(amp):.3f}")

    print("\nThe QFT concentrates amplitude at multiples of N/period!")
    print("This is key to quantum period finding in Shor's algorithm")


def demonstrate_qft_vs_classical_dft():
    """Compare QFT with classical DFT."""

    print("\n" + "=" * 70)
    print("QFT vs CLASSICAL DFT COMPARISON")
    print("=" * 70)

    print("\n--- Comparing 3-qubit QFT with 8-point Classical DFT ---")

    qft = QuantumFourierTransform(3)

    # Create a simple input state |3⟩ = |011⟩
    input_classical = [0] * 8
    input_classical[3] = 1

    # Classical DFT
    output_classical = qft.classical_dft(input_classical)

    # Quantum QFT
    result = qft.demonstrate_qft_on_basis_state(3)
    output_quantum = result["amplitudes"]

    print("\nInput: |011⟩ (position 3 = 1, rest = 0)")
    print("\nOutput comparison:")
    print("State | Classical DFT        | Quantum QFT         | Match?")
    print("------|---------------------|---------------------|-------")

    for i in range(8):
        classical_amp = output_classical[i]
        quantum_amp = output_quantum[i]
        match = np.allclose(classical_amp, quantum_amp)
        print(
            f" {i:3d}  | {abs(classical_amp):6.3f} ∠{cmath.phase(classical_amp) / np.pi:6.2f}π | "
            f"{abs(quantum_amp):6.3f} ∠{cmath.phase(quantum_amp) / np.pi:6.2f}π | {'✓' if match else '✗'}"
        )

    print("\nQFT implements the same transformation as classical DFT")
    print("but can do it for superposition states in O(n²) gates!")


def demonstrate_qft_applications():
    """Show applications of QFT in quantum algorithms."""

    print("\n" + "=" * 70)
    print("QFT APPLICATIONS IN QUANTUM ALGORITHMS")
    print("=" * 70)

    print("\n1. QUANTUM PHASE ESTIMATION")
    print("   - Estimates eigenvalues of unitary operators")
    print("   - Core subroutine in many quantum algorithms")
    print("   - Uses QFT to extract phase information")

    print("\n2. SHOR'S ALGORITHM")
    print("   - Factors integers in polynomial time")
    print("   - Uses QFT for period finding")
    print("   - QFT converts periodicity to measurable peaks")

    print("\n3. QUANTUM SIGNAL PROCESSING")
    print("   - Analyzes frequency components")
    print("   - Quantum advantage for certain signals")
    print("   - QFT provides exponential speedup")

    print("\n4. SOLVING LINEAR SYSTEMS (HHL)")
    print("   - Uses QFT in eigenvalue inversion")
    print("   - Exponential speedup for sparse matrices")

    # Simple phase estimation example
    print("\n--- Mini Example: Phase Kickback with QFT ---")

    # Create circuit showing phase estimation concept
    circuit = QuantumCircuit(3, 2)

    # Prepare control qubits in superposition
    circuit.h([0, 1])

    # Controlled rotations (simplified)
    angle = np.pi / 4
    circuit.cp(angle, 0, 2)
    circuit.cp(2 * angle, 1, 2)

    # Apply inverse QFT to extract phase
    qft = QuantumFourierTransform(2)
    iqft = qft.create_inverse_qft_circuit(2)
    circuit.append(iqft, [0, 1])

    # Measure
    circuit.measure([0, 1], [0, 1])

    print(circuit.draw(output="text"))
    print("\nThis circuit estimates the phase using QFT")


def interactive_qft():
    """Interactive QFT demonstration."""

    print("\n" + "=" * 70)
    print("INTERACTIVE QFT EXPLORATION")
    print("=" * 70)

    while True:
        try:
            n_input = input("\nNumber of qubits for QFT (2-5, or 'quit'): ").strip()

            if n_input.lower() == "quit":
                break

            n_qubits = int(n_input)
            if not (2 <= n_qubits <= 5):
                print("Please enter between 2 and 5 qubits")
                continue

            qft = QuantumFourierTransform(n_qubits)
            N = 2**n_qubits

            print(f"\n{n_qubits}-qubit QFT (transforms {N} basis states)")

            # Show circuit?
            if n_qubits <= 4:
                show_circuit = input("Show QFT circuit? (y/n): ").lower() == "y"
                if show_circuit:
                    circuit = qft.create_qft_circuit()
                    print("\nQFT Circuit:")
                    print(circuit.draw(output="text"))

            # Choose input state
            print(f"\nEnter input state (0-{N - 1}) or 'all' to see all basis states:")
            state_input = input().strip()

            if state_input.lower() == "all":
                print("\nQFT on all basis states:")
                print("Input | Output peaks (|amplitude| > 0.1)")
                print("------|--------------------------------")

                for state in range(N):
                    result = qft.demonstrate_qft_on_basis_state(state)
                    peaks = []
                    for i, amp in enumerate(result["amplitudes"]):
                        if abs(amp) > 0.1:
                            peaks.append(f"{i}({abs(amp):.2f})")

                    binary = format(state, f"0{n_qubits}b")
                    print(f"|{binary}⟩ | {', '.join(peaks)}")

            else:
                try:
                    state = int(state_input)
                    if not (0 <= state < N):
                        print(f"State must be between 0 and {N - 1}")
                        continue

                    result = qft.demonstrate_qft_on_basis_state(state)

                    print(f"\nInput state: |{result['binary']}⟩ (decimal: {state})")
                    print("\nOutput amplitudes:")
                    print("State   | Amplitude  | Phase")
                    print("--------|------------|------------")

                    for i, amp in enumerate(result["amplitudes"]):
                        if abs(amp) > 0.001:  # Only show non-negligible amplitudes
                            binary = format(i, f"0{n_qubits}b")
                            magnitude = abs(amp)
                            phase = cmath.phase(amp)
                            print(
                                f"|{binary}⟩ | {magnitude:10.3f} | {phase / np.pi:6.2f}π"
                            )

                    # Check if uniform superposition
                    if all(
                        abs(abs(amp) - 1 / np.sqrt(N)) < 0.001
                        for amp in result["amplitudes"]
                    ):
                        print("\nNote: QFT of |0⟩ gives uniform superposition!")

                except ValueError:
                    print("Please enter a valid integer")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_qft_basics()
    demonstrate_qft_patterns()
    demonstrate_qft_vs_classical_dft()
    demonstrate_qft_applications()

    # Interactive exploration
    interactive_qft()
