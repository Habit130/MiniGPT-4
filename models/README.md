# 模型文件放置说明

模型权重体积较大且受各自许可证约束，因此不纳入 Git 仓库。启动前请准备成以下结构：

```text
models/
├── checkpoint_0.pth
├── eva_vit_g.pth
├── vicuna-7b/
│   ├── config.json
│   ├── tokenizer.model
│   ├── pytorch_model-00001-of-00002.bin
│   └── ...
└── opus-mt-en-zh/
    ├── config.json
    ├── source.spm
    ├── target.spm
    ├── pytorch_model.bin
    └── ...
```

## 所需模型

1. `vicuna-7b/`
   - 来源：[Vision-CAIR/vicuna-7b](https://huggingface.co/Vision-CAIR/vicuna-7b)
   - 用途：MiniGPT-v2 的语言模型。
2. `checkpoint_0.pth`
   - 来源：本项目训练得到的作物病害微调权重。
   - 用途：加载视觉投影层与 LoRA 适配权重。
3. `eva_vit_g.pth`
   - 来源：[BLIP2 EVA ViT-G](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth)
   - 用途：视觉编码器基础权重。
4. `opus-mt-en-zh/`
   - 来源：[Helsinki-NLP/opus-mt-en-zh](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh)
   - 用途：当视觉语言模型返回英文时，离线转换为简体中文。

可使用 Git LFS 下载 Hugging Face 模型：

```bash
git lfs install
git clone https://huggingface.co/Vision-CAIR/vicuna-7b models/vicuna-7b
git clone https://huggingface.co/Helsinki-NLP/opus-mt-en-zh models/opus-mt-en-zh
```

请在下载和分发前分别阅读并遵守上述模型的许可证及使用条款。
