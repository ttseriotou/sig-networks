# SigNetworks: Sequential Path Signature Networks

SigNetworks (`sig-networks`) is a PyTorch package for training and evaluating
neural networks for longitudinal NLP classification tasks. `sig-networks` is a
library that applies models first developed in
[Sequential Path Signature Networks for Personalised Longitudinal Language Modeling](https://aclanthology.org/2023.findings-acl.310/)
by Tseriotou et al. (2023) which presented a novel extension of neural
sequential models using the notion of path signatures from rough path theory.

## Installation

SigNetworks is available on PyPI and can be installed with pip:

```bash
pip install sig_networks
```

### Signatory/Torch

SigNetworks depends on the
[`patrick-kidger/signatory`](https://github.com/patrick-kidger/signatory)
library for differentiable computation of path signatures/log-signatures in
PyTorch. Please see the
[signatory documentation](https://signatory.readthedocs.io/en/latest/) for
installation instructions of the signatory library.

## Usage

The key components in the _signature-window_ model s presented in (see
[Sequential Path Signature Networks for Personalised Longitudinal Language Modeling](https://aclanthology.org/2023.findings-acl.310/)
for full details) are written as PyTorch modules which can be used in a modular
fashion. The key components are:

- The Signature Window Network Units (SWNUs):
  [`sig_networks.SWNU`](src/sig_networks/swnu.py)
- The Signature Window (Multihead-)Attention Units (SWMHAUs):
  [`sig_networks.SWMHAU`](src/sig_networks/swmhau.py)
- The SWNU-Network model:
  [`sig_networks.SWNUNetwork`](src/sig_networks/swnu_network.py)
- The SWMHAU-Network model:
  [`sig_networks.SWMHAUNetwork`](src/sig_networks/swmhau_network.py)
- The SeqSigNet model:
  [`sig_networks.SeqSigNet`](src/sig_networks/seqsignet_bilstm.py)
- The SeqSigNet-Attention-Encoder model:
  [`sig_networks.SeqSigNetAttentionEncoder`](src/sig_networks/seqsignet_attention.py)
- The SeqSigNet-Attention-BiLSTM model:
  [`sig_networks.SeqSigNetAttentionBiLSTM`](src/sig_networks/seqsignet_attention_bilstm.py)

```python
...
```

## Pre-commit and linters

To take advantage of `pre-commit`, which will automatically format your code and
run some basic checks before you commit:

```
pip install pre-commit  # or brew install pre-commit on macOS
pre-commit install  # will install a pre-commit hook into the git repo
```

After doing this, each time you commit, some linters will be applied to format
the codebase. You can also/alternatively run `pre-commit run --all-files` to run
the checks.
