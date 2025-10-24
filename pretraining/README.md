# Code for SLL Pretraining

We’ve made the **Self-Supervised Learning (SLL) pretraining code** publicly available for your use and adaptation.  
The base implementation uses **L1 loss**, but it can easily be extended with **contrastive loss** — simply uncomment the relevant lines in the script.

> 💡 **Tip:**  
> The code is modular and well-structured, allowing you to integrate it directly into your own pipelines with minimal effort.

### Using the Code

Feel free to **copy, modify, and adapt** the pretraining scripts for your specific tasks or datasets.  
They’re designed to serve as flexible templates for your self-supervised representation learning experiments.

---

# Pretraining Losses — L1

Below are visual summaries of the **L1 reconstruction losses** during pretraining for both the **ResNet Autoencoder** and **ViT Autoencoder** architectures.

---

## 🧠 ResNet Autoencoder Pretraining

<div align="center">
  <img src="resnet_autoenc_pretraining.png" alt="ResNet Autoencoder Pretraining" width="60%" />
  <p><em>Figure 1. L1 loss curve during ResNet autoencoder pretraining.</em></p>
</div>

---

## 🔍 ViT Autoencoder Pretraining

<div align="center">
  <img src="vit_autoenc_pretraining.png" alt="ViT Autoencoder Pretraining" width="60%" />
  <p><em>Figure 2. L1 loss curve during ViT autoencoder pretraining.</em></p>
</div>

---

### Summary

These visualizations demonstrate how each architecture learns to minimize the **L1 reconstruction loss** over training epochs.  
You can use similar setups to pretrain models on your own datasets before fine-tuning them for downstream tasks.

