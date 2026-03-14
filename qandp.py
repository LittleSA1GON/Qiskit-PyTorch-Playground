import numpy as np
import torch

from qiskit import QuantumCircuit
from qiskit.circuit.library import real_amplitudes, zz_feature_map
from qiskit.primitives import StatevectorEstimator as Estimator

from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

# Reproducibility
algorithm_globals.random_seed = 42
torch.manual_seed(42)

# Tiny synthetic dataset: label is sign(x0 + x1)
num_inputs = 2
num_samples = 30
X = 2 * algorithm_globals.random.random([num_samples, num_inputs]) - 1
y = (np.sum(X, axis=1, keepdims=True) >= 0).astype(np.float32)
y = 2 * y - 1  # map {0,1} -> {-1,+1}

X_ = torch.tensor(X, dtype=torch.float32)
y_ = torch.tensor(y, dtype=torch.float32)

# Quantum circuit = feature map + ansatz
feature_map = zz_feature_map(num_inputs)
ansatz = real_amplitudes(num_inputs)

qc = QuantumCircuit(num_inputs)
qc.compose(feature_map, inplace=True)
qc.compose(ansatz, inplace=True)

# QNN + TorchConnector
estimator = Estimator()
qnn = EstimatorQNN(
    circuit=qc,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
    estimator=estimator,
)

initial_weights = 0.1 * (2 * algorithm_globals.random.random(qnn.num_weights) - 1)
model = TorchConnector(qnn, initial_weights=initial_weights)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.LBFGS(model.parameters(), max_iter=25)

def closure():
    optimizer.zero_grad()
    pred = model(X_)
    loss = loss_fn(pred, y_)
    loss.backward()
    return loss

model.train()
final_loss = optimizer.step(closure)

with torch.no_grad():
    pred_sign = model(X_).sign()
    acc = (pred_sign == y_).float().mean().item()

print("Final loss:", float(final_loss))
print("Train accuracy:", acc)
print("Example prediction:", float(model(X_[0]).detach()))