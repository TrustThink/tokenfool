# tokenfool

Widely recognized methods for producing adversarial examples (e.g., PGD, FGSM) were developed and popularized in the context of convolutional neural network (CNN)-based image models. These attacks operate in continuous input space and rely on end-to-end differentiability; in practice, they are typically applied as pixel-level perturbations to induce misclassification. Existing open-source adversarial toolkits such as [Foolbox](https://github.com/bethgelab/foolbox), [CleverHans](https://github.com/cleverhans-lab/cleverhans), and [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox) consolidate these implementations, enabling users to benchmark robustness and evaluate defenses through standardized interfaces. 

In recent years, transformer-based models have emerged as a dominant alternative to CNNs in computer vision. Architectures such as ViT, DeiT, Swin, and DETR replace convolutional feature extraction with patch tokenization and self-attention mechanisms. These models are now widely deployed across classification, detection, and segmentation tasks. Like CNNs, vision transformers are vulnerable to adversarial examples under standard white-box gradient attacks. However, their token-based representation and global self-attention introduce different structural sensitivities compared to convolutional architectures. In addition to conventional pixel-space perturbations, transformers are susceptible to structured patch-level attacks, token/embedding-space perturbations, and attacks that exploit attention distributions or object query representations in detection models. 

## About

TokenFool is an open-source toolkit for adversarial attacks on image transformers. It is modular and extensible, enabling straightforward integration of new attack algorithms. Attacks operate under gradient-based white-box assumptions and integrate directly with standard PyTorch training and evaluation workflows. Users can support their own models by implementing the interface contract expected by the attacks. This keeps attack code separate from provider-specific or model-specific wiring and makes it possible to extend TokenFool beyond the built-in adapters.

## Current Status

TokenFool currently includes:
- a core `tokenfool` package for attack implementations
- an adapter/interface layer for transformer classifier integrations
- example usage notebooks 
- automated test coverage
- repository CI workflows

Current attack coverage includes:
- **Patch-Fool** ([paper](https://arxiv.org/abs/2203.08392))
- **Adaptive Token Tuning (ATT)** ([paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/24f8dd1b8f154f1ee0d7a59e368eccf3-Abstract-Conference.html))
- **Pay No Attention (PNA) + PatchOut** ([paper](https://arxiv.org/pdf/2109.04176))

Current model support is focused on:
- **timm and huggingface ViT/DeiT classifier implementations**
- **user-defined models that implement the required interface contract via a custom adapter**

## Next Steps

Planned expansion areas include:
- additional transformer-specific attack methods
- broader model-family and provider coverage
- support for non-classification transformer settings through new interfaces and adapters


## TokenFool Architecture

![Architecture diagram](architecture.png)

The library is structured as a layered system: attacks depend on stable interface contracts rather than directly on model implementations. Concrete adapters bridge those interfaces to supported model implementations, while custom adapters provide the path for integrating user-owned models without changing attack logic.

## Development

1. Clone the repository   
(**External contributors**: fork the repository first, then clone your fork)
```bash
git clone https://github.com/TrustThink/tokenfool.git
cd tokenfool
```
2. (Recommended) Create a virtual environment

3. Install the project in editable mode with development dependencies
```bash
pip install -e ".[dev]"
```

4. Make changes on a new branch, including tests were appropriate

5. Run tests
```bash
pytest
```

6. Run lint checks
```bash
ruff check
```

7. Open a Pull Request

Repository checks also run through GitHub Actions.
