# SSL_Pretraining

## Self-Supervised Learning  
Revealing the Power of Vision in Multimodal Tasks through Pretraining

## Pretraining  
SSL Pretraining

## Strategic Finetuning  
For stroke relapse detection


## Data Description

- **Data types:**
  - 3D CTAs
  - Tabular data including:
    - Age
    - Gender
    - CHD (Coronary Heart Disease)
    - PAD (Peripheral Artery Disease)

- **Note:**
  The data cannot be published due to legal restrictions.
	
## Dataset Overview

## Data Availability

The datasets used for the **SLL pretraining** and **Fine-Tuning os Stroke Relapse task** are **not publicly available** due to institutional and licensing restrictions.
However, all experiments were conducted on **standard open-source benchmarks**, and equivalent data can be obtained from publicly accessible repositories.

If you wish to use the code to your own datasets, you can:

- Replace the dataset loaders with your custom data paths.
- Follow the same preprocessing pipeline provided in the training scripts.
- Ensure your dataset follows a similar format (images + metadata where applicable).

> 🗂️ **Note:**
> While the original datasets cannot be redistributed, we encourage the use of **open equivalents**.


## Pretraining
- **Total samples:** 491 3D CTAs from 491 patients (at initial stroke event)
- **Split:**
  - **Training:** 393
  - **Validation:** 98

## Finetuning
- **Total samples:** 119 image–tabular pairs from 119 patients
- **Split:**
  - **Training:** 95
  - **Testing:** 24

## Code

All code related to **SLL pretraining** is made **fully available** in this repository.  
This includes implementations for the **ResNet Autoencoder** and **ViT Autoencoder** used in the L1-based self-supervised pretraining stage.

The code for **fine-tuning and architecture training** (ResNetMLP and ViTMLP) is **partially available**, with some modules withheld due to institutional and legal restrictions.

> ⚙️ **Summary:**
> - ✅ Pretraining code — *fully released*
> - ⚠️ Fine-tuning & architecture code — *partially released*
> - 💡 Both models are easily extendable and can be integrated into your own workflows or datasets.



## Citation

If you use this repository, or if our project has inspired or supported your research in any way,
please consider citing it to acknowledge the effort behind its development 🙌.

### 📚 How to Cite

You can cite the repository as follows:

> **Project:** 
> **Authors:** [Christian Gapp], et al.
> **Year:** 2025
> **Repository:** [https://github.com/ChristianGappGit/SSL_Pretraining](https://github.com/ChristianGappGit/SSL_Pretraining)

### BibTeX Entry

```bibtex
@misc{Gapp2025_SSL_Stroke,
  title        = {Vision’s Potential Unlocked: How Pretraining and Strategic Fine-tuning Improve Stroke Relapse Prediction},
  author       = {Christian Gapp and Elias Tappeiner and Martin Welk and Karl Fritscher and Stephanie Mangesius and Constantin Eisenschink and Astrid E. Grams and Elke R. Gizewski and Philipp Deisl and Michael Knoflach and Rainer Schubert},
  year         = {2025},
  howpublished = {\url{https://github.com/ChristianGappGit/SSL_Pretraining}},
  note         = {Available at GitHub. Pretraining and model architecture resources for SSL.}
}
