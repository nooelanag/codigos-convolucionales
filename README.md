# Convolutional Coding and Viterbi Decoding

End-to-end simulation of a digital communications system with convolutional coding, AWGN channel, and Kalman filter-based interference cancellation.

---

## Description

This project implements an end-to-end communications system using a **rate R = 1/2 convolutional code** with generator matrix G(D) = [D² + 1, D² + D + 1] and **Viterbi algorithm decoding** with hard decision.

The system allows you to:

- Evaluate the performance (BER vs E_b/N₀) of the convolutional code compared to uncoded transmission over an AWGN channel with BPSK modulation.
- Validate the implementation by comparing simulated results against theoretical error probability curves.
- Recover a hidden message transmitted through a channel with structured interference, using a Kalman filter to estimate and cancel the interference prior to decoding.

## Key Features

- **Generic convolutional encoder**: accepts any generator matrix in polynomial form, implemented using `numpy.convolve`.
- **Configurable AWGN channel**: simulates BPSK transmission with noise variance adjusted to the code rate, returning both soft values and hard decision output (BSC).
- **Viterbi decoder**: full implementation of the algorithm with Hamming distance metric for the (2,1,3) code with 4 states.
- **Kalman filter**: estimates the state of a linear interfering system from indirect observations, enabling interference cancellation before decoding.
- **Monte Carlo simulation**: statistical performance evaluation by averaging over multiple independent frames.
- **Plot generation**: BER curves on a logarithmic scale comparing coded, uncoded, and theoretical values.
- **ASCII message decoding**: complete pipeline from interference cancellation to bit-to-text conversion.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Steps

1. **Clone the repository**:

```bash
git clone https://github.com/dalouc/codigos-convolucionales.git
cd codigos-convolucionales
```

2. **Install dependencies**:

```bash
pip install numpy scipy matplotlib
```

3. **Verify the installation**:

```bash
python -c "import numpy; import scipy; import matplotlib; print('Dependencies OK')"
```

## Usage

### Full Execution

From the `src/` directory, run the main program:

```bash
cd src
python main.py
```

This automatically performs:

1. Monte Carlo BER simulation for E_b/N₀ from 0 to 10 dB (uncoded and convolutionally coded systems).
2. Generation of `ber_curve.png` (channel BER vs. theoretical) and `ber_curve_e2e.png` (end-to-end BER after Viterbi decoding).
3. Interference cancellation and hidden message decoding (requires `observations.txt` and `data_rx.txt` in the working directory).

### Individual Module Usage

Each module can be used independently:

```python
import numpy as np
from conv_encoding import conv_encoding
from awgn_channel import awgn_channel
from viterbi_decode_101_111 import viterbi_decode_101_111

# Encode
info_bits = np.array([1, 0, 1, 1, 0])
coded = conv_encoding(info_bits, [[1,0,1], [1,1,1]])

# Transmit through AWGN channel (Eb/N0 = 5 dB, rate R = 0.5)
rx_soft, rx_hard = awgn_channel(coded, EbN0_dB=5.0, R=0.5)

# Decode
decoded = viterbi_decode_101_111(rx_hard)
print(f"Original: {info_bits}")
print(f"Decoded:  {decoded[:len(info_bits)]}")
```

### Adjusting Simulation Parameters

The main parameters can be modified in `main.py`:

```python
EbN0_dB_range = np.arange(0, 11, 1)   # Eb/N0 range in dB
n_bits = 1000                         # Information bits per frame
n_frames = 100                        # Number of frames per simulation point
```

> For reliable statistics at BER ≤ 10⁻⁴, it is recommended to increase `n_bits` and `n_frames` (e.g., 10000 bits × 500 frames), at the cost of longer execution time.

## Project Structure

```
.
├── src/
│   ├── main.py                      # Main program: simulation and plots
│   ├── conv_encoding.py             # Generic convolutional encoder
│   ├── awgn_channel.py              # AWGN channel simulation + hard decision
│   ├── viterbi_decode_101_111.py    # Viterbi decoder for G(D)=[D²+1, D²+D+1]
│   ├── interf_cancel.py             # Kalman filter + interference cancellation
│   ├── convert2char.py              # Bit-to-ASCII character conversion
│   ├── observations.txt             # Observations y[k] from the interfering system (*)
│   └── data_rx.txt                  # Received signal r[k] = s[k] + x[k] (*)
├── .gitignore
└── README.md
```

### Module Descriptions

| File | Purpose |
|---|---|
| `main.py` | Orchestrates the Monte Carlo simulation, generates BER curves, and runs interference cancellation. |
| `conv_encoding.py` | Encodes a bit sequence using `numpy.convolve` with each generator polynomial (mod 2) and interleaves the outputs. |
| `awgn_channel.py` | Maps bits to BPSK, adds Gaussian noise with σ² = 1/(2·R·E_b/N₀), and applies hard decision. |
| `viterbi_decode_101_111.py` | Implements the Viterbi algorithm with Hamming distance metric for the (2,1,3) code with 4 states. |
| `interf_cancel.py` | Contains the Kalman filter and the interference cancellation function that estimates x[k], cleans the signal, decodes, and converts to ASCII. |
| `convert2char.py` | Groups bits into 8-bit blocks and converts them to ASCII characters. |

## Configuration

### Convolutional Code Parameters

The code used has the following properties:

| Parameter | Value |
|---|---|
| Generator matrix | G(D) = [D² + 1, D² + D + 1] |
| Polynomials (binary) | g₁ = [1, 0, 1], g₂ = [1, 1, 1] |
| Rate | R = 1/2 |
| Constraint length | K = 3 |
| Number of states | 2^(K-1) = 4 |
| Free distance | d_free = 5 |

### Interference System Parameters

Defined in the assignment and hardcoded in `interf_cancel.py`:

```
Transition matrix:    A = [[0.9, 0.12], [0.08, 0.85]]
Observation matrix:   B = [[-0.12, 0.15], [0.4, -0.15]]
Initial state:        x[0] = [0, 0]
Process noise:        u[k] ~ N(0, 0.15² · I₂)
Observation noise:    v[k] ~ N(0, 0.15² · I₂)
```

### Channel Model

The AWGN channel uses BPSK modulation with the mapping `0 → -1`, `1 → +1`. The noise variance is adjusted according to the code rate:

- Uncoded (R = 1): P_e = Q(√(2·E_b/N₀))
- Coded (R = 0.5): P_e = Q(√(2·R·E_b/N₀))

## Expected Results

Running `main.py` produces:

1. **Console output**: table with simulated BER for each E_b/N₀ point and the decoded message.
2. **`ber_curve.png`**: simulated channel BER overlaid with theoretical curves, validating the implementation.
3. **`ber_curve_e2e.png`**: end-to-end BER (after Viterbi decoding) showing the ~3 dB coding gain at BER = 10⁻⁴.

## Authors

[Noel Andolz Aguado](https://github.com/nooelanag) and [Daniel Lozano Uceda](https://github.com/dalouc)
