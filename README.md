# Quantum Circuit Analysis & Generation Framework

A practical framework for **analyzing** quantum circuit properties and **generating** new circuits conditioned on target objectives. The system combines a **Decision Transformer (DT)** with **graph neural network (GNN)** embeddings to predict circuit properties and to autoregressively generate ansätze under hardware constraints.

---

## Overview

- **Property Analysis** (`Ansatz_Data_ver2`)
  - **Entanglement**: Meyer–Wallach entropy
  - **Expressibility**: KL-divergence w.r.t. Haar reference
  - **(Inverse) Fidelity**: inverse-circuit measurement (probability of \(|0\rangle^{\otimes n}\))
  - Supports IBM Quantum backends and local simulators
- **Circuit Generation** (`OAT_Model`)
  - Decision Transformer with **G/R/S/A** tokenization (Guide, Reward-To-Go, State, Action)
  - GNN-based circuit embeddings (connectivity & gate structure)
  - Causal masking for strict autoregressive inference
  - Optional conditioning on target property ranges

---

## Repository Structure

~~~text
Kaist/
├─ OAT_Model/                     # Decision Transformer (generation) + predictors
│  ├─ src/
│  │  ├─ models/                 # Transformer, GNN, predictors
│  │  ├─ training/               # Training pipelines & loops
│  │  ├─ data/                   # Dataset definitions & loaders
│  │  └─ utils/                  # Common utilities
│  ├─ raw_data/                  # Example data stubs / pointers
│  └─ examples/                  # Inference & usage examples
├─ Ansatz_Data_ver2/             # Property analysis tools
│  ├─ core/                      # Core measurement routines
│  ├─ execution/                 # IBM Quantum execution handlers
│  └─ expressibility/            # Expressibility metrics
├─ quantumcommon/                # Shared circuit interfaces/helpers
└─ validation_results/           # Plots and validation artifacts
~~~

---

## Installation

~~~bash
git clone <your-repo-url>
cd Kaist

# Core Python deps (pin versions as needed)
pip install torch torchvision
pip install torch-geometric
pip install qiskit matplotlib numpy pandas

# (Optional) IBM Quantum token
cp OAT_Model/.env.example OAT_Model/.env
# Edit OAT_Model/.env and set: IBM_TOKEN=<your-token>
~~~

> **Note:** Simply use 'install_pytorch_geometric.sh' for copy and run to install torch-geometric.

---

## Quick Start

### 1) Run Quantum Circuit Generation with 3 Properties measurement(Fidelity, Entanglement, Expressibility)

~~~bash
cd Ansatz_Data_ver2

# Batch analysis
python main.py

# Statistical validation / sanity checks
python run_statistical_validation.py
~~~


### 2) Train a property predictor

~~~bash
cd OAT_Model/src

python unified_experiment_runner.py \
  --data_path ../raw_data/merged_data.json \
  --epoch 50 \
  --experiment_type property_prediction

# With GPU
python unified_experiment_runner.py \
  --data_path ../raw_data/merged_data.json \
  --epoch 50 \
  --experiment_type property_prediction \
  --device cuda
~~~

### 3) Train the Decision Transformer (generation)

~~~bash
cd OAT_Model

python train_decision_transformer.py \
  --model_size small \
  --epochs 100 \
  --learning_rate 1e-4 \
  --device cuda

# Autoregressive circuit generation (uses the latest checkpoint)
python examples/autoregressive_circuit_generation.py
~~~

---

## Data

The framework expects circuit datasets with structure and measured properties:

- **Structure:** gate sequence, qubit connectivity, parameters  
- **Properties:** entanglement (Meyer–Wallach), expressibility (KL), (inverse) fidelity  
- **Backends:** real hardware (e.g., IBM) and simulators  
- **Typical scales:** 2–20 qubits, depth 1–50

For private datasets, keep raw files outside the repo and point `--data_path` to your prepared JSON.

---

## Features

- **Conditioned Generation:** DT generates gate tokens under target property ranges  
- **GNN Embeddings:** captures topology and register-level structure  
- **Causal Inference Path:** strict masked decoding for reproducible evaluation  
- **Multi-Property Prediction:** joint training for entanglement, expressibility, (inverse) fidelity  
- **Validation Utilities:** correlation checks, statistical tests, and plotting helpers  
- **Performance:** minibatch pipelines and optional AMP (mixed precision)

---

## Results & Reproducibility

This repo provides training/evaluation scripts and plotting utilities to reproduce:

- property-prediction learning curves,  
- generation diagnostics,  
- validation plots on your dataset.

When reporting results, include:

- dataset description (size, qubits, depth, hardware/simulator),  
- training config (epochs, batch size, optimizer, seeds),  
- exact commit/version.

---

## Contributing

Contributions are welcome. Potential directions:

- additional property metrics and validators,  
- improved circuit encodings and tokenizers,  
- backend integrations (more IBM devices, emulators, or other providers),  
- minimal, self-contained examples and tutorials.

Please open an issue before large changes.

---

## License

This project is intended for research use. If you use the code or ideas, please provide appropriate attribution.

---

## Citation

If this framework assists your research, consider citing:

~~~bibtex
@misc{quantum_circuit_oat_framework,
  title  = {OPTIMIZING QUANTUM CIRCUIT DESIGN THROUGH DECISION
TRANSFORMERS: IN REAL QUANTUM COMPUTER},
  author = {Junyoung Jung},
  year   = {2025},
  note   = {https://github.com/junyoung7727/qforge}
}
~~~

---

## Related Materials

See `quantum_analysis_plots/` and any `paper_figure*.pdf` files for figures and analysis scripts corresponding to your experiments.
