from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler

# Bell state circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

# Add measurements (returns a measured copy)
qc_measured = qc.measure_all(inplace=False)

sampler = StatevectorSampler()
job = sampler.run([qc_measured], shots=1000)
result = job.result()

counts = result[0].data["meas"].get_counts()
print("Counts:", counts)
