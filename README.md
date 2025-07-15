<img src="./h-net.png" width="450px"></img>

## H-Net Dynamic Chunking (wip)

Implementation of the dynamic chunking mechanism in [H-net](https://arxiv.org/abs/2507.07955) by Hwang et al. of Carnegie Mellon

## Install

```shell
$ pip install h-net-dynamic-chunking
```

## Usage

```python
import torch
from h_net_dynamic_chunking.h_net_dynamic_chunking import DynamicChunkingDownsampler

downsampler = DynamicChunkingDownsampler(512)

tokens = torch.randn(3, 1024, 512).requires_grad_()

downsampled, upsample_fn, *_ = downsampler(tokens)

assert upsample_fn(downsampled).shape == tokens.shape
```

## Citations

```bibtex
@misc{hwang2025dynamicchunkingendtoendhierarchical,
    title   = {Dynamic Chunking for End-to-End Hierarchical Sequence Modeling},
    author  = {Sukjun Hwang and Brandon Wang and Albert Gu},
    year    = {2025},
    eprint  = {2507.07955},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2507.07955},
}
```
