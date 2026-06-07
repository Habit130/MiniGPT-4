# CropGuard: A MiniGPT-v2 Crop Disease Visual Diagnosis Platform

**English** | [简体中文](README_zh-CN.md)

CropGuard is a Chinese-language visual question answering web application for images of crop leaves, stems, and fruit. It is built on MiniGPT-v2 and loads LoRA and visual projection weights fine-tuned for crop disease scenarios.

This repository contains the complete web interface, inference code, configuration, environment definition, launch scripts, and eight crop example images. Model weights are excluded from Git because of their size and licensing requirements. Users only need to place the weights as documented, run one command to create the environment, and run one command to launch the interface.

> This project is intended for research, education, and assisted assessment only. It does not replace field diagnosis by agricultural professionals or pesticide label requirements.

## Features

- Fully Chinese crop disease diagnosis interface
- Upload crop images and ask follow-up questions
- Health diagnosis, symptom analysis, cause assessment, control advice, and crop identification modes
- Eight built-in real crop examples with no external dataset dependency
- Automatic detection and loading of Linear or MLP visual projection weights
- LoRA rank and critical-weight validation to prevent silently skipped weights
- Local English-to-Chinese translation when the vision-language model answers in English
- Fully local model loading and offline inference after all weights are prepared

## Interface Examples

| Tomato Late Blight | Apple Cedar Rust | Rice Blast | Healthy Bell Pepper |
| --- | --- | --- | --- |
| ![](assets/examples/tomato_late_blight.jpg) | ![](assets/examples/apple_cedar_rust.jpg) | ![](assets/examples/rice_blast.jpg) | ![](assets/examples/bell_pepper_healthy.jpg) |

## Requirements

- Windows 10/11 or a common Linux distribution
- NVIDIA GPU with at least 12 GB VRAM recommended
- NVIDIA driver compatible with CUDA 11.8
- At least 24 GB system memory recommended
- Miniconda or Anaconda
- Git LFS for downloading large Hugging Face models

The project has been verified with Python 3.9 and PyTorch 2.6.0 + CUDA 11.8.

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Habit130/MiniGPT-4.git
cd MiniGPT-4
```

### 2. Prepare Model Weights

The final directory layout must be:

```text
MiniGPT-4/
└── models/
    ├── checkpoint_0.pth
    ├── eva_vit_g.pth
    ├── vicuna-7b/
    │   ├── config.json
    │   ├── tokenizer.model
    │   └── ...
    └── opus-mt-en-zh/
        ├── config.json
        ├── source.spm
        ├── target.spm
        └── ...
```

Download the two Hugging Face models:

```bash
git lfs install
git clone https://huggingface.co/Vision-CAIR/vicuna-7b models/vicuna-7b
git clone https://huggingface.co/Helsinki-NLP/opus-mt-en-zh models/opus-mt-en-zh
```

Download EVA ViT-G.

Windows PowerShell:

```powershell
Invoke-WebRequest "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth" -OutFile "models/eva_vit_g.pth"
```

Linux:

```bash
curl -L "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth" -o models/eva_vit_g.pth
```

Finally, place the crop disease fine-tuned `checkpoint_0.pth` in `models/`. See [models/README.md](models/README.md) for more details.

### 3. Build the Environment with One Command

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
```

Linux:

```bash
bash scripts/setup_env.sh
```

The script creates a Conda environment named `minigpt-crop`, then validates PyTorch, CUDA, and the core dependencies.

### 4. Launch the Web Interface with One Command

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_web.ps1
```

Linux:

```bash
bash scripts/start_web.sh
```

Model loading usually takes 1-3 minutes. After startup, the browser opens automatically at:

```text
http://127.0.0.1:7861/
```

Press `Ctrl+C` to stop the service.

## Launch Options

Windows examples:

```powershell
# Use GPU 1 and port 8080
powershell -ExecutionPolicy Bypass -File .\scripts\start_web.ps1 -GpuId 1 -Port 8080

# Allow access from other devices on the local network
powershell -ExecutionPolicy Bypass -File .\scripts\start_web.ps1 -ServerName 0.0.0.0

# Do not open the browser automatically
powershell -ExecutionPolicy Bypass -File .\scripts\start_web.ps1 -NoBrowser
```

On Linux, use environment variables:

```bash
MINIGPT_GPU_ID=1 MINIGPT_PORT=8080 MINIGPT_SERVER_NAME=0.0.0.0 bash scripts/start_web.sh
```

You can also run Python directly:

```bash
conda run -n minigpt-crop --no-capture-output python -u demo_v2.py \
  --cfg-path eval_configs/minigptv2_checkpoint_0_eval.yaml \
  --gpu-id 0 \
  --server-name 127.0.0.1 \
  --server-port 7861 \
  --inbrowser
```

## Project Structure

```text
.
├── assets/examples/                 # Built-in crop examples
├── eval_configs/
│   └── minigptv2_checkpoint_0_eval.yaml
├── minigpt4/
│   └── models/projection.py         # Projection and weight compatibility logic
├── models/                          # User-provided weights, excluded from Git
├── scripts/
│   ├── setup_env.ps1
│   ├── setup_env.sh
│   ├── start_web.ps1
│   └── start_web.sh
├── tests/test_projection.py
├── demo_v2.py                       # Chinese Gradio web entry point
└── environment.yml
```

## Verification

Run the projection and weight compatibility tests:

```bash
conda run -n minigpt-crop python tests/test_projection.py -v
```

Quickly validate the key Python files:

```bash
conda run -n minigpt-crop python -m py_compile demo_v2.py minigpt4/models/projection.py minigpt4/models/minigpt_v2.py
```

## Troubleshooting

### Missing Model Files

The launch script checks all four model components. Confirm that every filename and directory level exactly matches [models/README.md](models/README.md). Avoid nesting an additional Hugging Face repository directory.

### CUDA Is Unavailable

Run:

```bash
conda run -n minigpt-crop python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

If the output is `False`, verify that your NVIDIA driver supports CUDA 11.8 and that you did not install a CPU-only PyTorch build.

### Out of GPU Memory

Close other GPU-intensive applications, keep `low_resource: true`, or use a GPU with more VRAM. The project loads Vicuna 7B in 8-bit mode by default.

### The Page Opens but Example Images Are Missing

Confirm that all eight JPG files exist in `assets/examples/`. The project no longer reads an external `dataset` directory.

## Model and Data Notes

- `checkpoint_0.pth` contains crop disease fine-tuning weights. It does not contain the complete Vicuna, EVA ViT-G, or translation model parameters.
- See [assets/examples/README.md](assets/examples/README.md) for the example image list and original classes.
- Publishers and users must independently confirm that the licenses of the base models, fine-tuned weights, and image data permit their intended use and distribution.
- Do not commit the weights under `models/` to a standard Git repository.

## Acknowledgements

This project is based on the following open-source work:

- [MiniGPT-4 / MiniGPT-v2](https://github.com/Vision-CAIR/MiniGPT-4)
- [BLIP-2](https://huggingface.co/docs/transformers/model_doc/blip-2)
- [Vicuna](https://github.com/lm-sys/FastChat)
- [LLaMA](https://github.com/facebookresearch/llama)
- [Helsinki-NLP OPUS-MT](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh)

MiniGPT-v2 paper:

```bibtex
@article{chen2023minigptv2,
  title={MiniGPT-v2: Large Language Model as a Unified Interface for Vision-Language Multi-task Learning},
  author={Chen, Jun and Zhu, Deyao and Shen, Xiaoqian and Li, Xiang and Liu, Zechun and Zhang, Pengchuan and Krishnamoorthi, Raghuraman and Chandra, Vikas and Xiong, Yunyang and Elhoseiny, Mohamed},
  journal={arXiv preprint arXiv:2310.09478},
  year={2023}
}
```

## License

The code follows [LICENSE.md](LICENSE.md) and [LICENSE_Lavis.md](LICENSE_Lavis.md) in this repository. Model weights and data resources remain subject to their respective licenses.
