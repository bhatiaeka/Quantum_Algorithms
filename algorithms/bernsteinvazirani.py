"""
Bernstein-Vazirani Algorithm Implementation

This quantum algorithm finds a hidden bit string 's' encoded in an oracle function
f(x) = s·x (mod 2) with just one quantum query, demonstrating quantum parallelism.
"""

# Install required packages:
# pip install qiskit qiskit-aer matplotlib
from typing import Dict
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram, circuit_drawer
import matplotlib.pyplot as plt


class BernsteinVaziraniQiskit:
    """Bernstein-Vazirani algorithm implementation using Qiskit."""

    def __init__(self, secret_string: str):
        """
        Initialize with a secret bit string.

        Args:
            secret_string: Binary string to be discovered (e.g., "101")
        """
        self.secret_string = secret_string
        self.n_qubits = len(secret_string)

    def create_oracle(self) -> QuantumCircuit:
        """
        Create the oracle circuit for the secret string.
        Oracle implements: |x⟩|y⟩ → |x⟩|y ⊕ (s·x)⟩

        Returns:
            Oracle as a QuantumCircuit
        """
        oracle = QuantumCircuit(self.n_qubits + 1, name="Oracle")

        # Apply CNOT for each bit that is 1 in the secret string
        for i, bit in enumerate(reversed(self.secret_string)):
            if bit == "1":
                oracle.cx(i, self.n_qubits)  # CNOT from qubit i to ancilla

        return oracle

    def create_circuit(self) -> QuantumCircuit:
        """
        Create the complete Bernstein-Vazirani circuit.

        Returns:
            Complete quantum circuit
        """
        # Create quantum and classical registers
        qreg = QuantumRegister(self.n_qubits, "q")
        ancilla = QuantumRegister(1, "ancilla")
        creg = ClassicalRegister(self.n_qubits, "c")

        # Create circuit
        circuit = QuantumCircuit(qreg, ancilla, creg)

        # Step 1: Initialize ancilla to |1⟩
        circuit.x(ancilla[0])

        # Step 2: Apply Hadamard to all qubits (including ancilla)
        circuit.h(qreg)
        circuit.h(ancilla)

        circuit.barrier()

        # Step 3: Apply oracle
        oracle = self.create_oracle()
        circuit.append(oracle, range(self.n_qubits + 1))

        circuit.barrier()

        # Step 4: Apply Hadamard to input qubits (not ancilla)
        circuit.h(qreg)

        circuit.barrier()

        # Step 5: Measure input qubits
        circuit.measure(qreg, creg)

        return circuit

    def run(self, shots: int = 1024, show_circuit: bool = True) -> Dict:
        """
        Run the Bernstein-Vazirani algorithm.

        Args:
            shots: Number of measurement shots
            show_circuit: Whether to display the circuit diagram

        Returns:
            Dictionary with results
        """
        # Create circuit
        circuit = self.create_circuit()

        # Show circuit if requested
        if show_circuit:
            print("\nQuantum Circuit:")
            print(circuit.draw(output="text"))

        # Run on simulator
        simulator = AerSimulator()
        compiled_circuit = transpile(circuit, simulator)
        job = simulator.run(compiled_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)

        # The result should be the secret string with high probability
        measured_string = max(counts, key=counts.get)

        return {
            "counts": counts,
            "measured": measured_string,
            "success": measured_string
            == self.secret_string[::-1],  # Reverse due to Qiskit's bit ordering
            "circuit": circuit,
        }

    def visualize_results(self, counts: Dict):
        """
        Visualize measurement results as a histogram.

        Args:
            counts: Measurement counts dictionary
        """
        # Create histogram
        fig = plot_histogram(
            counts, title=f"BV Algorithm Results\nSecret: {self.secret_string}"
        )
        plt.show()
        return fig


def demonstrate_bernstein_vazirani():
    """Demonstrate the Bernstein-Vazirani algorithm with examples."""

    print("=" * 70)
    print("BERNSTEIN-VAZIRANI ALGORITHM DEMONSTRATION WITH QISKIT")
    print("=" * 70)

    # Example 1: Simple 3-bit string
    print("\n--- Example 1: 3-bit secret string ---")
    secret = "101"
    bv = BernsteinVaziraniQiskit(secret)

    print(f"Secret string: {secret}")
    results = bv.run(shots=1024, show_circuit=True)

    print(f"\nMeasurement results:")
    for state, count in sorted(
        results["counts"].items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  |{state}⟩: {count} times ({count / 1024 * 100:.1f}%)")

    print(f"\nMost frequent measurement: {results['measured']}")
    print(f"Expected (reversed): {secret[::-1]}")  # Qiskit uses reverse bit ordering
    print(f"Success: {results['success']}")

    # Example 2: Larger string
    print("\n--- Example 2: 5-bit secret string ---")
    secret = "11010"
    bv = BernsteinVaziraniQiskit(secret)

    print(f"Secret string: {secret}")
    results = bv.run(
        shots=1024, show_circuit=False
    )  # Don't show circuit for larger example

    print(f"\nMeasurement results (top 3):")
    sorted_counts = sorted(results["counts"].items(), key=lambda x: x[1], reverse=True)
    for state, count in sorted_counts[:3]:
        print(f"  |{state}⟩: {count} times ({count / 1024 * 100:.1f}%)")

    print(f"\nMost frequent measurement: {results['measured']}")
    print(f"Expected (reversed): {secret[::-1]}")
    print(f"Success: {results['success']}")

    # Example 3: Perfect success demonstration
    print("\n--- Example 3: Why it works perfectly ---")
    print("The algorithm achieves 100% success rate (in ideal conditions) because:")
    print("1. It uses quantum parallelism to query all possible inputs simultaneously")
    print("2. The Hadamard gates create and then undo superposition")
    print(
        "3. Quantum interference ensures only the secret string has non-zero amplitude"
    )
    print("\nClassical algorithm: n queries needed")
    print("Quantum algorithm: 1 query needed")
    print(f"Speedup for {len(secret)}-bit string: {len(secret)}x")

    # Example 4: Circuit depth analysis
    print("\n--- Example 4: Circuit Analysis ---")
    for n in [2, 3, 4, 5, 6]:
        bv = BernsteinVaziraniQiskit("1" * n)
        circuit = bv.create_circuit()
        print(
            f"{n} qubits: Depth = {circuit.depth()}, Gates = {sum(dict(circuit.count_ops()).values())}"
        )

    return results


def compare_with_classical():
    """Compare quantum vs classical approach."""

    print("\n" + "=" * 70)
    print("QUANTUM vs CLASSICAL COMPARISON")
    print("=" * 70)

    def classical_find_secret(oracle_function, n_bits):
        """Classical approach: query each bit position."""
        secret = []
        queries = 0

        for i in range(n_bits):
            # Create input with only bit i set to 1
            x = [0] * n_bits
            x[i] = 1

            # Query oracle
            result = oracle_function(x)
            secret.append(str(result))
            queries += 1

        return "".join(secret), queries

    # Example oracle for classical algorithm
    def make_classical_oracle(secret_string):
        def oracle(x):
            return sum(int(s) * xi for s, xi in zip(secret_string, x)) % 2

        return oracle

    # Compare for different string lengths
    print("\n| Bits | Classical Queries | Quantum Queries | Speedup |")
    print("|------|------------------|-----------------|---------|")

    for n in [4, 8, 16, 32, 64]:
        classical_queries = n
        quantum_queries = 1
        speedup = classical_queries / quantum_queries
        print(
            f"| {n:4d} | {classical_queries:16d} | {quantum_queries:15d} | {speedup:6.0f}x |"
        )

    print("\nThe quantum advantage is LINEAR in the number of bits!")
    print("This demonstrates exponential speedup for the oracle query complexity.")


def interactive_mode():
    """Interactive mode for testing custom secret strings."""

    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)

    while True:
        try:
            secret = input(
                "\nEnter secret binary string (e.g., 1101) or 'quit' to exit: "
            ).strip()

            if secret.lower() == "quit":
                break

            # Validate input
            if not secret:
                print("Please enter a non-empty string.")
                continue

            if not all(c in "01" for c in secret):
                print("Please enter only 0s and 1s.")
                continue

            if len(secret) > 10:
                print("For visualization purposes, please use 10 bits or fewer.")
                continue

            # Run algorithm
            print(f"\nRunning Bernstein-Vazirani for secret: {secret}")
            bv = BernsteinVaziraniQiskit(secret)

            # Ask for options
            show_circuit = input("Show circuit diagram? (y/n): ").lower() == "y"

            results = bv.run(shots=1024, show_circuit=show_circuit)

            print(f"\nResults:")
            print(f"Secret string:    {secret}")
            print(
                f"Measured string:  {results['measured'][::-1]}"
            )  # Reverse for display
            print(f"Success:          {results['success']}")

            if results["success"]:
                print("✓ Algorithm successfully found the secret string!")
            else:
                print("✗ Unexpected result (this is rare with ideal simulation)")

            # Show measurement distribution
            show_dist = input("\nShow measurement distribution? (y/n): ").lower() == "y"
            if show_dist:
                sorted_counts = sorted(
                    results["counts"].items(), key=lambda x: x[1], reverse=True
                )
                print("\nMeasurement distribution:")
                for state, count in sorted_counts[:5]:
                    bar = "█" * int(count / 1024 * 50)
                    print(f"|{state}⟩: {bar} {count / 1024 * 100:.1f}%")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    # Run demonstration
    demonstrate_bernstein_vazirani()

    # Show classical comparison
    compare_with_classical()

    # Run interactive mode
    interactive_mode()
