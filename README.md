# Hybrid NIDS: Parallel ML-Powered Network Intrusion Detection

A hybrid Network Intrusion Detection System that pairs a **deep-learning DDoS classifier** trained in Python/PyTorch with a **high-performance C++ detection engine**. The C++ engine runs the model through **LibTorch (TorchScript)** and processes packets in parallel with **OpenMP**.

The idea: train where it's convenient (Python), and detect where it's fast (multithreaded C++).

## Highlights

- **~96% accuracy** (5-fold stratified cross-validation) on CIC-IDS2017 DDoS traffic, and **0.97 weighted F1** on the held-out test set (45k flows).
- **~3.5× higher throughput** from the parallel OpenMP engine vs. the sequential baseline: ~171k vs ~47k packets/sec at 50k packets, with per-packet latency dropping from ~21 µs to ~5.9 µs.
- Scaled to **5 million packets** in benchmark runs.

## Architecture

```
          ┌──────────────── Python (training) ─────────────────┐
CIC-IDS2017 → cleaning → variance / correlation / importance → StandardScaler
  CSV flows   (inf/NaN)   based feature selection               + LabelEncoder
                                  │
                                  ▼
                 PyTorch classifier (MLP / Transformer encoder)
                 AdamW · cosine LR · early stopping · 5-fold CV
                                  │  torch.jit.script
                                  ▼
                        TorchScript model (.pt)
          └────────────────────────────────────────────────────┘
                                  │
          ┌──────────────── C++ (detection) ───────────────────┐
  packet stream → feature extraction/normalisation → LibTorch inference
                 OpenMP parallel-for (dynamic scheduling),
                 thread-local counters merged in critical sections
                                  │
                                  ▼
          DDoS / Benign verdicts + throughput & latency benchmarks
          └────────────────────────────────────────────────────┘
```

### Models explored
- **AdvancedDDoSDetector**: a deep MLP (128 → 64 → 32) with BatchNorm and progressive dropout.
- **NetworkIntrusionTransformer**: a custom Transformer encoder with multi-head attention and positional encoding over flow features.

## Results

**Classification (CIC-IDS2017, Friday DDoS capture)**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| BENIGN | 1.00 | 0.92 | 0.96 |
| DDoS | 0.94 | 1.00 | 0.97 |
| **Weighted avg** | **0.97** | **0.97** | **0.97** |

**Detection engine (sequential vs. parallel)**

| Packets | Sequential throughput | Parallel throughput | Speed-up |
|---|---|---|---|
| 10,000 | ~44k pkt/s | ~155k pkt/s | ~3.5× |
| 50,000 | ~47k pkt/s | ~171k pkt/s | ~3.6× |

## Repository structure

```
Code/
├── ML Model/
│   ├── train_network_ids.ipynb   # Preprocessing, feature selection, training, CV, TorchScript export
│   ├── best_ddos_model.pth       # Trained weights
│   └── *.png                     # Confusion matrix, feature importance/correlation, training curves
├── Packet Capturing/
│   ├── sequential_nids.cpp       # Single-threaded baseline detector
│   ├── parallel_nids.cpp         # OpenMP + LibTorch parallel detector
│   ├── CMakeLists.txt, build.sh  # Build configuration
│   └── build/                    # Benchmark result files
└── model/                        # Fitted scaler and label encoder
```

## Build & run

**Requirements:** CMake ≥ 3.10, LLVM clang with OpenMP (`brew install llvm libomp`), and LibTorch (`brew install pytorch` or the [official build](https://pytorch.org/get-started/locally/)).

```bash
cd "Code/Packet Capturing"
./build.sh                                  # builds parallel and sequential detectors under build/
./build/parallel/parallel_nids              # run the parallel detector
./build/sequential/sequential_nids          # run the sequential baseline
```

Point `model_path` in the C++ sources to your exported TorchScript model.

**Training:** open `Code/ML Model/train_network_ids.ipynb`, set the path to the CIC-IDS2017 `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` file, and run all cells.

## Tech stack

C++14 · OpenMP · LibTorch/TorchScript · CMake · Python · PyTorch · scikit-learn · pandas · Matplotlib/Seaborn
