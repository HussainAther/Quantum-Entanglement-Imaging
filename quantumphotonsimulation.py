from qiskit import QuantumCircuit, execute, Aer
import numpy as np

def entangled_photon_simulation():
    # Create a quantum circuit with 2 qubits
    qc = QuantumCircuit(2)
    qc.h(0)  # Apply Hadamard gate on the first qubit
    qc.cx(0, 1)  # Apply CNOT gate on the second qubit

    # Simulate the quantum circuit
    simulator = Aer.get_backend('statevector_simulator')
    result = execute(qc, simulator).result()

    # Get the statevector (wave function)
    statevector = result.get_statevector(qc)
    return statevector

if __name__ == "__main__":
    result = entangled_photon_simulation()
    print(f"Entangled State: {result}")

