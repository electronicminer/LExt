

# LExt: PyTorch Reproduction of "Listen to Extract"

<div align="center">
  <img src="[https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)" alt="PyTorch">
  <img src="[https://img.shields.io/badge/Dataset-WSJ0--2mix-blue?style=flat-square](https://img.shields.io/badge/Dataset-WSJ0--2mix-blue?style=flat-square)" alt="Dataset">
  <img src="[https://img.shields.io/badge/Task-Target_Speaker_Extraction-success?style=flat-square](https://img.shields.io/badge/Task-Target_Speaker_Extraction-success?style=flat-square)" alt="Task">
</div>

## 📖 Introduction
This repository contains an unofficial PyTorch reproduction of the paper **[Listen to Extract: Onset-Prompted Target Speaker Extraction](https://arxiv.org/abs/2505.05114)** (Shen et al., 2025). 

Unlike traditional Target Speaker Extraction (TSE) methods that rely on complex cross-attention or explicit speaker embeddings, LExt proposes an elegant and simple approach: concatenating the enrollment utterance directly at the front of the mixture in the time domain, separated by an all-zero "glue signal." This creates an artificial speech onset, guiding the network (e.g., TF-GridNet) to extract the target speaker effectively.


## ✨ Key Features & Improvements
In this reproduction, several engineering optimizations were made to ensure training stability and enhance the purity of the onset prompt:

* **Modern VAD Integration**: Replaced the traditional Kaldi-based SAD with the enterprise-grade deep learning model **Silero VAD**. This significantly improves the robustness against background noise and ensures a highly purified enrollment onset, maximizing the effectiveness of the concatenation mechanism.
* **Optimization Strategy**: Implemented a **Cosine Annealing** learning rate scheduler (decaying from 1e-3 to 1e-6 over 50 epochs) to replace standard plateau-based decay, ensuring smooth convergence in the later stages of training.
* **Uncompromised Numerical Stability**: Given the high spatiotemporal complexity of TF-GridNet, training was conducted on a single H100 (80GB) GPU. To strictly prevent regression accuracy loss, AMP was disabled (full FP32 training), and robust gradient clipping (max_norm=5.0) was applied to prevent exploding gradients during the unfolding of the sequential network.

## 📊 Current Results (Stage 1)
Evaluated on the standard WSJ0-2mix (min, 8kHz) test set containing 3,000 unseen mixtures. 

| Backbone | Params | Training Setup | Epochs | SI-SDRi (dB) |
| :--- | :--- | :--- | :---: | :---: |
| LExt (TF-GridNetV2) | ~14M | Single H100 80GB, FP32, Batch Size 4 | 22 / 50 | **21.24** |

*(Note: The model is currently at epoch 22 and the validation loss is still smoothly decreasing. Further training is expected to push the performance closer to the paper's reported limit of 24.1 dB).*

## 🛠️ Project Structure
```text
LExt/
├── decode_wsj0.py         # Script to decode .wv1 files to standard .wav using FFmpeg
├── merge_wsj0.py          # Script to merge WSJ0 CD directories
├── modern_vad.py          # Wrapper for the Silero VAD engine
├── offline_dataset.py     # Custom PyTorch Dataset for dynamic mixture & enrollment concatenation
├── model_and_loss.py      # TF-GridNetV2 wrapper and Negative SI-SDR loss implementation
├── train.py               # Initial training script
├── train_v2.py            # Resuming training script with Cosine Annealing LR Scheduler
├── evaluate.py            # Evaluation script for calculating SI-SDRi on the test set
├── inference.py           # Script to perform TSE on custom audio files
└── listen_test.py         # Script to generate audio demos from the test set
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
git clone https://github.com/electronicminer/LExt.git
cd LExt
pip install torch torchaudio torchmetrics tqdm
```
*(Silero VAD will be automatically downloaded via torch.hub upon first execution).*

### 2. Data Preparation
1.  Use `merge_wsj0.py` to consolidate the CSR-1 CD directories.
2.  Use `decode_wsj0.py` to convert the `.wv1` files to standard `.wav` format.
3.  Use the standard WSJ0-2mix scripts (not included here) to generate the mixed datasets. Ensure your directory is structured as `tr`, `cv`, and `tt`.

### 3. Training
To start training from scratch:
```bash
python train.py
```
To resume training (e.g., from epoch 15) with the aligned Cosine Annealing scheduler:
```bash
python train_v2.py
```

### 4. Evaluation & Inference
Evaluate the model's SI-SDRi on the complete test set:
```bash
python evaluate.py
```

Extract target speech using your own custom audio and enrollment:
```bash
python inference.py
```

## 🔗 Citation & Acknowledgments
If you find this reproduction helpful, please consider citing the original paper:
```bibtex
@article{shen2025listen,
  title={Listen to Extract: Onset-Prompted Target Speaker Extraction},
  author={Shen, Pengjie and Chen, Kangrui and He, Shulin and others},
  journal={arXiv preprint arXiv:2505.05114},
  year={2025}
}
```
* Thanks to the authors of LExt for their innovative and simple approach to TSE.
* Thanks to the [Silero VAD](https://github.com/snakers4/silero-vad) team for their excellent voice activity detection model.

# LExt: “Listen to Extract” 论文的 PyTorch 复现

<div align="center">
  <img src="[https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)" alt="PyTorch">
  <img src="[https://img.shields.io/badge/Dataset-WSJ0--2mix-blue?style=flat-square](https://img.shields.io/badge/Dataset-WSJ0--2mix-blue?style=flat-square)" alt="Dataset">
  <img src="[https://img.shields.io/badge/Task-Target_Speaker_Extraction-success?style=flat-square](https://img.shields.io/badge/Task-Target_Speaker_Extraction-success?style=flat-square)" alt="Task">
</div>

## 📖 项目简介
本项目包含了对论文 **[Listen to Extract: Onset-Prompted Target Speaker Extraction](https://arxiv.org/abs/2505.05114)** (Shen et al., 2025) 的非官方 PyTorch 端到端复现。

不同于传统的依赖复杂交叉注意力机制或显式说话人特征（如 x-vector）的目标语音提取（TSE）方法，LExt 提出了一种优雅且极简的方案：直接在波形时域上，将注册语音（Enrollment）拼接在混合语音（Mixture）的最前端，中间用全零的“胶水信号（Glue Signal）”隔开。这种方式人为构造了一个语音“起始点（Onset）”，从而极其有效地引导网络（如 TF-GridNet）精准提取出目标说话人的声音。



## ✨ 核心特性与工程优化
在本次复现中，为了保证端到端训练的绝对稳定性，并进一步提升前端引导信号的纯净度，我在此项目中引入了以下关键工程优化：

* **集成企业级 VAD 模型**：使用基于深度学习的企业级 **Silero VAD** 替换了原论文中基于 Kaldi 的传统 SAD 工具。这极大提升了模型在处理复杂底噪和微弱气声时的边界裁剪鲁棒性，确保了拼接前端声纹特征的高纯度，从数据源头最大化了时域拼接机制的有效性。
* **引入高级学习率调度**：使用了**余弦退火（Cosine Annealing）**学习率调度器，设定在 50 个 Epoch 内将学习率从 1e-3 平滑且连续地衰减至 1e-6。相比传统的阶梯式降速，该策略有效避免了训练后期的梯度震荡，确保了模型向极限性能的平滑逼近。
* **无损的数值稳定性保障**：考虑到 TF-GridNet 极高的时空复杂度，本项目基于单张 H100 (80GB) 显卡进行极限压榨。为了绝对避免回归任务的精度受损，全程**禁用半精度混合训练（AMP），坚持使用纯 FP32 单精度计算**，并配合阈值为 5.0 的梯度裁剪（Gradient Clipping）机制，完美规避了时序网络展开早期的梯度爆炸风险。

## 📊 阶段性评估结果
模型在标准的 WSJ0-2mix (min, 8kHz) 测试集（包含 3000 条未见混合语音）上进行了性能评估。

| 主干网络 | 参数量 | 训练硬件与配置 | 当前迭代轮数 | 平均 SI-SDRi (dB) |
| :--- | :--- | :--- | :---: | :---: |
| LExt (TF-GridNetV2) | ~14M | 单卡 H100 80GB, FP32, Batch Size 4 | 22 / 50 | **21.24** |

*(注：模型目前仅训练至第 22 轮，验证集 Loss 仍在平滑健康地下降。随着后续训练周期的拉长，预计提取性能将进一步逼近论文原报告的 24.1 dB 极限水平。)*

## 🛠️ 项目代码结构
```text
LExt/
├── decode_wsj0.py         # 调用 FFmpeg 将原生 .wv1 格式解码为标准 .wav 的脚本
├── merge_wsj0.py          # 自动化合并 WSJ0 光盘目录结构的脚本
├── modern_vad.py          # 封装现代 Silero VAD 引擎的工具类
├── offline_dataset.py     # 自定义 PyTorch Dataset (实现动态时域拼接与对齐)
├── model_and_loss.py      # TF-GridNetV2 主干网络封装及 Negative SI-SDR 损失函数
├── train.py               # 初始模型训练脚本
├── train_v2.py            # 带有余弦退火策略的接力训练脚本
├── evaluate.py            # 在完整测试集上计算平均 SI-SDRi 提升的评估脚本
├── inference.py           # 支持输入自定义音频文件进行目标语音提取的推理脚本
└── listen_test.py         # 用于从测试集中快速抽取音频生成试听 Demo 的脚本
```

## 🚀 快速开始

### 1. 环境配置
```bash
git clone https://github.com/electronicminer/LExt.git
cd LExt
pip install torch torchaudio torchmetrics tqdm
```
*(注意：首次运行涉及到 VAD 处理的代码时，系统会自动从 torch.hub 下载 Silero VAD 权重)。*

### 2. 数据准备
1.  运行 `merge_wsj0.py` 整合 CSR-1 原始 CD 目录。
2.  运行 `decode_wsj0.py` 将 `.wv1` 文件批量无损转换为 `.wav` 格式。
3.  请使用标准的 WSJ0-2mix 开源数据生成脚本（本项目未包含）生成混合数据集，并确保文件目录按 `tr`, `cv`, `tt` 结构放置。

### 3. 模型训练
如果您希望从零开始训练模型：
```bash
python train.py
```
如果您希望加载已有权重，并使用对齐好的余弦退火策略接力训练（例如从第 15 轮开始）：
```bash
python train_v2.py
```

### 4. 评估与推理
在完整测试集（Test Set）上评估模型的 SI-SDRi 指标：
```bash
python evaluate.py
```

使用您自己录制的混合音频和目标说话人纯净音频进行分离推理：
```bash
python inference.py
```

## 🔗 引用与致谢
如果您觉得本项目对您的研究有所帮助，请考虑引用原论文：
```bibtex
@article{shen2025listen,
  title={Listen to Extract: Onset-Prompted Target Speaker Extraction},
  author={Shen, Pengjie and Chen, Kangrui and He, Shulin and others},
  journal={arXiv preprint arXiv:2505.05114},
  year={2025}
}
```
* 特别感谢 LExt 论文原作者团队在目标语音提取领域提出的这一极具启发性的简洁思路。
* 感谢 [Silero VAD](https://github.com/snakers4/silero-vad) 团队开源的出色语音活动检测模型。
