"""
Grover's Algorithm Implementation using Qiskit

A clean implementation of Grover's quantum search algorithm using Qiskit,
demonstrating quadratic speedup for searching unsorted databases.
"""

# Install required packages:
# pip install qiskit qiskit-aer matplotlib

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram, circuit_drawer
from qiskit.circuit.library import MCXGate
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import math


class GroverQiskit:
    """Grover's algorithm implementation using Qiskit."""

    def __init__(self, n_qubits: int, marked_items: List[int]):
        """
        Initialize Grover's algorithm.

        Args:
            n_qubits: Number of qubits (searches 2^n items)
            marked_items: List of indices to mark (e.g., [5, 10])
        """
        self.n_qubits = n_qubits
        self.n_items = 2**n_qubits
        self.marked_items = marked_items

        # Calculate optimal number of iterations
        self.n_iterations = self.calculate_optimal_iterations()

    def calculate_optimal_iterations(self) -> int:
        """
        Calculate optimal number of Grover iterations.

        Returns:
            Optimal number of iterations
        """
        if len(self.marked_items) == 0:
            return 0

        # For M marked items out of N total items:
        # Optimal iterations ≈ π/4 * sqrt(N/M)
        ratio = self.n_items / len(self.marked_items)
        iterations = int(np.floor(np.pi / 4 * np.sqrt(ratio)))

        return max(1, iterations)

    def create_oracle(self) -> QuantumCircuit:
        """
        Create the oracle that marks the target items.
        Flips the phase of marked items: |x⟩ → -|x⟩ if x is marked

        Returns:
            Oracle circuit
        """
        oracle = QuantumCircuit(self.n_qubits, name="Oracle")

        for marked in self.marked_items:
            # Convert marked index to binary
            marked_binary = format(marked, f"0{self.n_qubits}b")

            # Add X gates for 0 bits (to convert to all 1s)
            for i, bit in enumerate(reversed(marked_binary)):
                if bit == "0":
                    oracle.x(i)

            # Multi-controlled Z gate (phase flip)
            if self.n_qubits == 1:
                oracle.z(0)
            elif self.n_qubits == 2:
                oracle.cz(0, 1)
            else:
                # Create multi-controlled Z using controlled phase
                oracle.h(self.n_qubits - 1)
                oracle.mcx(list(range(self.n_qubits - 1)), self.n_qubits - 1)
                oracle.h(self.n_qubits - 1)

            # Undo X gates
            for i, bit in enumerate(reversed(marked_binary)):
                if bit == "0":
                    oracle.x(i)

        return oracle

    def create_diffuser(self) -> QuantumCircuit:
        """
        Create the diffusion operator (inversion about average).

        Returns:
            Diffuser circuit
        """
        diffuser = QuantumCircuit(self.n_qubits, name="Diffuser")

        # Apply H gates
        diffuser.h(range(self.n_qubits))

        # Apply X gates
        diffuser.x(range(self.n_qubits))

        # Multi-controlled Z gate
        if self.n_qubits == 1:
            diffuser.z(0)
        elif self.n_qubits == 2:
            diffuser.cz(0, 1)
        else:
            # Multi-controlled Z
            diffuser.h(self.n_qubits - 1)
            diffuser.mcx(list(range(self.n_qubits - 1)), self.n_qubits - 1)
            diffuser.h(self.n_qubits - 1)

        # Apply X gates
        diffuser.x(range(self.n_qubits))

        # Apply H gates
        diffuser.h(range(self.n_qubits))

        return diffuser

    def create_circuit(self, iterations: int = None) -> QuantumCircuit:
        """
        Create the complete Grover circuit.

        Args:
            iterations: Number of Grover iterations (None for optimal)

        Returns:
            Complete quantum circuit
        """
        if iterations is None:
            iterations = self.n_iterations

        # Create registers
        qreg = QuantumRegister(self.n_qubits, "q")
        creg = ClassicalRegister(self.n_qubits, "c")
        circuit = QuantumCircuit(qreg, creg)

        # Step 1: Initialize with Hadamard gates
        circuit.h(range(self.n_qubits))

        # Step 2: Apply Grover operator (Oracle + Diffuser) iterations times
        oracle = self.create_oracle()
        diffuser = self.create_diffuser()

        for i in range(iterations):
            circuit.barrier()
            circuit.append(oracle, range(self.n_qubits))
            circuit.append(diffuser, range(self.n_qubits))

        # Step 3: Measure
        circuit.barrier()
        circuit.measure(qreg, creg)

        return circuit

    def run(
        self, shots: int = 1024, iterations: int = None, show_circuit: bool = True
    ) -> Dict:
        """
        Run Grover's algorithm.

        Args:
            shots: Number of measurement shots
            iterations: Number of iterations (None for optimal)
            show_circuit: Whether to display the circuit

        Returns:
            Dictionary with results
        """
        # Create circuit
        circuit = self.create_circuit(iterations)

        # Show circuit if requested
        if show_circuit and self.n_qubits <= 4:  # Only show for small circuits
            print("\nQuantum Circuit:")
            print(circuit.draw(output="text"))

        # Run simulation
        simulator = AerSimulator()
        compiled_circuit = transpile(circuit, simulator)
        job = simulator.run(compiled_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)

        # Check success
        marked_binary = [format(m, f"0{self.n_qubits}b") for m in self.marked_items]

        # Reverse bit order for Qiskit
        marked_binary_reversed = [b[::-1] for b in marked_binary]

        # Calculate success probability
        success_count = sum(counts.get(m, 0) for m in marked_binary_reversed)
        success_prob = success_count / shots

        return {
            "counts": counts,
            "success_probability": success_prob,
            "iterations_used": iterations if iterations else self.n_iterations,
            "marked_items": self.marked_items,
            "circuit": circuit,
        }

    def visualize_results(self, counts: Dict, title: str = None):
        """
        Visualize measurement results.

        Args:
            counts: Measurement counts
            title: Plot title
        """
        if title is None:
            title = f"Grover's Algorithm Results\nMarked: {self.marked_items}"

        # Highlight marked items in the plot
        marked_binary = [
            format(m, f"0{self.n_qubits}b")[::-1] for m in self.marked_items
        ]

        # Color bars
        colors = [
            "red" if state in marked_binary else "blue" for state in counts.keys()
        ]

        fig = plot_histogram(counts, title=title, color=colors)
        plt.show()
        return fig


def demonstrate_grover():
    """Demonstrate Grover's algorithm with various examples."""

    print("=" * 70)
    print("GROVER'S ALGORITHM DEMONSTRATION WITH QISKIT")
    print("=" * 70)

    # Example 1: Small database, single marked item
    print("\n--- Example 1: 4-item database (2 qubits), 1 marked ---")
    grover = GroverQiskit(n_qubits=2, marked_items=[3])

    print(f"Database size: {grover.n_items} items")
    print(f"Marked item: {grover.marked_items[0]} (binary: 11)")
    print(f"Optimal iterations: {grover.n_iterations}")

    results = grover.run(shots=1024, show_circuit=True)

    print(f"\nResults after {results['iterations_used']} iteration(s):")
    sorted_counts = sorted(results["counts"].items(), key=lambda x: x[1], reverse=True)
    for state, count in sorted_counts[:3]:
        decimal = int(state[::-1], 2)  # Convert to decimal
        marked = "← MARKED" if decimal in grover.marked_items else ""
        print(
            f"  |{state}⟩ (item {decimal}): {count} times ({count / 1024 * 100:.1f}%) {marked}"
        )

    print(f"\nSuccess probability: {results['success_probability']:.1%}")

    # Example 2: Larger database, multiple marked items
    print("\n--- Example 2: 16-item database (4 qubits), 2 marked ---")
    grover = GroverQiskit(n_qubits=4, marked_items=[5, 10])

    print(f"Database size: {grover.n_items} items")
    print(f"Marked items: {grover.marked_items}")
    print(f"Optimal iterations: {grover.n_iterations}")

    results = grover.run(shots=1024, show_circuit=False)

    print(f"\nTop measurement results after {results['iterations_used']} iteration(s):")
    sorted_counts = sorted(results["counts"].items(), key=lambda x: x[1], reverse=True)
    for state, count in sorted_counts[:5]:
        decimal = int(state[::-1], 2)
        marked = "← MARKED" if decimal in grover.marked_items else ""
        print(
            f"  |{state}⟩ (item {decimal:2d}): {count:3d} times ({count / 1024 * 100:5.1f}%) {marked}"
        )

    print(f"\nSuccess probability: {results['success_probability']:.1%}")

    # Example 3: Effect of iteration count
    print("\n--- Example 3: Effect of Iteration Count ---")
    print("8-item database (3 qubits), 1 marked item")

    grover = GroverQiskit(n_qubits=3, marked_items=[5])
    optimal = grover.n_iterations

    print(f"\nOptimal iterations: {optimal}")
    print("\nIteration | Success Probability")
    print("-" * 32)

    for iters in range(0, optimal + 3):
        results = grover.run(shots=1024, iterations=iters, show_circuit=False)
        print(f"    {iters:2d}    |     {results['success_probability']:6.1%}")

    print("\nNote: Too many iterations decreases success probability!")

    # Example 4: Speedup demonstration
    print("\n--- Example 4: Quantum Speedup ---")
    print("\n| Qubits | Database Size | Classical (avg) | Quantum | Speedup |")
    print("|--------|---------------|-----------------|---------|---------|")

    for n in [3, 4, 5, 6, 8]:
        db_size = 2**n
        classical = db_size // 2  # Average case
        grover_test = GroverQiskit(n_qubits=n, marked_items=[0])
        quantum = grover_test.n_iterations
        speedup = classical / quantum

        print(
            f"|   {n:2d}   |      {db_size:3d}      |       {classical:3d}       |    {quantum:2d}   |  {speedup:5.1f}x |"
        )

    print("\nGrover provides quadratic speedup: O(√N) vs O(N)")


def amplitude_evolution_demo():
    """Show how amplitudes evolve during Grover iterations."""

    print("\n" + "=" * 70)
    print("AMPLITUDE EVOLUTION VISUALIZATION")
    print("=" * 70)

    print("\nTracking probability of finding marked item over iterations:")
    print("(8-item database, 1 marked)")

    grover = GroverQiskit(n_qubits=3, marked_items=[6])

    # Create visual representation
    iterations_to_test = range(0, 6)
    probabilities = []

    for iters in iterations_to_test:
        results = grover.run(shots=2048, iterations=iters, show_circuit=False)
        probabilities.append(results["success_probability"])

    # ASCII bar chart
    print("\n Iter | Probability | Visualization")
    print("------|-------------|" + "-" * 50)

    for iters, prob in zip(iterations_to_test, probabilities):
        bar = "█" * int(prob * 50)
        marker = " ← OPTIMAL" if iters == grover.n_iterations else ""
        print(f"  {iters:2d}  |   {prob:6.1%}    | {bar}{marker}")

    print("\nThe amplitude amplification is clearly visible!")


def interactive_grover():
    """Interactive mode for testing Grover's algorithm."""

    print("\n" + "=" * 70)
    print("INTERACTIVE GROVER'S SEARCH")
    print("=" * 70)

    while True:
        try:
            # Get number of qubits
            n_qubits = input("\nNumber of qubits (2-5, or 'quit' to exit): ").strip()

            if n_qubits.lower() == "quit":
                break

            n_qubits = int(n_qubits)
            if not (2 <= n_qubits <= 5):
                print("Please enter a number between 2 and 5.")
                continue

            db_size = 2**n_qubits
            print(f"Database size: {db_size} items (indices 0 to {db_size - 1})")

            # Get marked items
            marked_input = input(
                "Enter marked item indices (comma-separated, e.g., 3,7): "
            ).strip()
            marked_items = []

            for item in marked_input.split(","):
                try:
                    idx = int(item.strip())
                    if 0 <= idx < db_size:
                        marked_items.append(idx)
                    else:
                        print(f"Warning: {idx} is out of range, skipping.")
                except ValueError:
                    print(f"Warning: '{item.strip()}' is not valid, skipping.")

            if not marked_items:
                print("No valid marked items. Please try again.")
                continue

            # Create and run algorithm
            print(f"\nSearching for items: {marked_items}")
            grover = GroverQiskit(n_qubits=n_qubits, marked_items=marked_items)

            print(f"Optimal iterations: {grover.n_iterations}")

            # Option to customize iterations
            custom = input("Use custom iteration count? (y/n): ").lower() == "y"
            iterations = None
            if custom:
                iterations = int(
                    input(
                        f"Enter number of iterations (optimal is {grover.n_iterations}): "
                    )
                )

            # Run algorithm
            results = grover.run(
                shots=1024, iterations=iterations, show_circuit=(n_qubits <= 3)
            )

            # Show results
            print(f"\n{'=' * 40}")
            print("RESULTS:")
            print(f"Iterations used: {results['iterations_used']}")
            print(f"Success probability: {results['success_probability']:.1%}")

            print("\nTop 5 measurements:")
            sorted_counts = sorted(
                results["counts"].items(), key=lambda x: x[1], reverse=True
            )
            for state, count in sorted_counts[:5]:
                decimal = int(state[::-1], 2)
                marked = "← MARKED" if decimal in marked_items else ""
                print(
                    f"  Item {decimal:2d} (|{state}⟩): {count:3d} times ({count / 1024 * 100:5.1f}%) {marked}"
                )

            # Speedup info
            classical_avg = db_size // 2
            speedup = classical_avg / results["iterations_used"]
            print(f"\nClassical average: {classical_avg} queries")
            print(f"Quantum: {results['iterations_used']} queries")
            print(f"Speedup: {speedup:.1f}x")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_grover()

    # Show amplitude evolution
    amplitude_evolution_demo()

    # Interactive mode
    interactive_grover()
