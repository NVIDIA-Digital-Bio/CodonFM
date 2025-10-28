# Model Overview

## Description:
CodonFM predicts masked codons in mRNA sequences from codon-level context to enable variant effect interpretation and codon optimization as part of NVIDIA’s CodonFM Encodon family. For this family of models we have 4 models. The first set of 3 models are with randomly masked tokens with 80 million, 600 million and 1 Billion parameter. The fourth model is with 1 Billion parameters but is trained with codon frequency aware masking. <br>

An additional set of accelerated checkpoints also available for use.

This model is ready for commercial/non-commercial use. <br>


## License/Terms of Use

Governing Terms: Use of this model is governed by the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).

## Deployment Geography:
Global <br>

## Use Case: <br>

- Optimized Expression and Stability for mRNA design: To design mRNAs with codon usage patterns that enhance translation efficiency, protein yield, and transcript stability across specific cell types and tissues.

- Variant Interpretation for pathogenicity: To identify and prioritize functional synonymous and missense variants in the context of diseases. <br>

## Release Date:  <br>
Github 10/27/2025 via https://github.com/NVIDIA-Digital-Bio/CodonFM <br>
Hugging Face 10/27/2025 via:
- Random Mask
    - https://huggingface.co/nvidia/NV-CodonFM-Encodon-1B-v1
    - https://huggingface.co/nvidia/NV-CodonFM-Encodon-600M-v1
    - https://huggingface.co/nvidia/NV-CodonFM-Encodon-80M-v1
    - https://huggingface.co/nvidia/NV-CodonFM-Encodon-TE-1B-v1
    - https://huggingface.co/nvidia/NV-CodonFM-Encodon-TE-600M-v1
    - https://huggingface.co/nvidia/NV-CodonFM-Encodon-TE-80M-v1
- Codon Frequency Aware Mask
    - https://huggingface.co/nvidia/NV-CodonFM-Encodon-Cdwt-1B-v1
    - https://huggingface.co/nvidia/NV-CodonFM-Encodon-TE-Cdwt-1B-v1 <br>

NGC 10/27/2025 via https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/models/nv_codonfm_encodon <br>



## Model Architecture:
The NVIDIA CodonFM Encodon family features **Transformer-based architectures** tailored for codon-level sequence modeling in mRNA.
Each model applies a **masked language modeling (MLM)** objective to predict masked codons from surrounding context of 2046 codons, enabling genome-scale codon optimization and synonymous variant interpretation.

| Model Name | Parameters |
|-------------|-------------:|
| Encodon-80M | 7.68 × 10⁷ |
| Encodon-600M | 6.09 × 10⁸ |
| Encodon-1B | 9.11 × 10⁸ |
| Encodon-Cdwt-1B | 9.11 × 10⁸ |
<br>

## Input: <br>
**Input Type(s):** Text (mRNA Sequence)  <br>
**Input Format:** fasta files converted to memmaps <br>
**Input Parameters:** 1D <br>
**Other Properties Related to Input:** mRNA sequence represented as a string of codons, of maximum length 2046. Longer sequences are automatically truncated to this length


## Output: <br>
**Output Type(s):** mRNA Sequence <br>
**Output Format:** Text <br>
**Output Parameters:** 2D <br>
**Other Properties Related to Output:** Numeric 2D tensor with float-point values representing probabilities of a given codon at a give position within the sequence <br>

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA’s hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions. <br>

## Software Integration:
**Runtime Engine(s):**
* PyTorch - 2.5.1


**Supported Hardware Microarchitecture Compatibility:** <br>
* NVIDIA Ampere <br>
* NVIDIA Hopper


**Preferred/Supported Operating System(s):**
* Linux <br>


The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment. <br>


## Model Version(s):
- NV-CodonFM-Encodon-80M-v1 <br>
- NV-CodonFM-Encodon-600M-v1 <br>
- NV-CodonFM-Encodon-1B-v1 <br>
- NV-CodonFM-Encodon-Cdwt-1B-v1 <br>
- NV-CodonFM-Encodon-TE-80M-v1 <br>
- NV-CodonFM-Encodon-TE-600M-v1 <br>
- NV-CodonFM-Encodon-TE-1B-v1 <br>
- NV-CodonFM-Encodon-TE-Cdwt-1B-v1 <br>


## Training, Testing, and Evaluation Datasets:

# Training Dataset:

**Link:** [RefSeq Data from NCBI](https://ftp.ncbi.nlm.nih.gov/genomes/refseq/)   <br>

**Data Modality** <br>
* Text (mRNA Sequencing data) <br>

**Properties:** Coding sequences from the NCBI RefSeq database (release 2024-04) were used for training. A total of >130M non-viral protein-coding sequences from >22,000 species were included, comprising >2,000 eukaryotes. Sequences not divisible by three or containing ambiguous bases were removed. Taxonomy-level deduplication using MMSeqs eliminated redundant entries, and coding sequences from bacteria pathogenic to humans were excluded. The resulting dataset was partitioned into nine species groups: primates, archaea, bacteria, fungi, invertebrate, plant, protozoa, non-primate mammals, and non-mammal vertebrates. Sequences were clustered by similarity and then split into training and validation sets with stratification across groups to ensure balanced representation.

Encodon models use codon-level tokenization, processing input sequences of up to 2,046 codons. Each model was trained using a masked language modeling (MLM) objective, where randomly masked codons were predicted from their context. The Encodon pretraining dataset was sorted based on sequence taxonomy to maintain species balance, and sequence subsets could be resampled dynamically. <br>

**Non-Audio, Image, Text Training Data Size:**  NCBI RefSeq genomes FTP directory currently contains over 395,000 genomes totaling approximately 3.3 terabases (Tb)

**Data Collection Method for all data Dataset:**
* Automatic/Sensors <br>

**Labeling Method by Dataset:**
* Not Applicable <br>


### Evaluation Datasets:

| Link | Properties  |
|------|--------------------------------------------------|
| [ClinVar Variant Interpretation](https://www.ncbi.nlm.nih.gov/clinvar/) | This task involves classifying genetic variants from ClinVar, a publicly available database that aggregates information about the clinical significance of human genetic variants, into pathogenic or benign categories based on their coding sequence context|
| [Denovo variant classification](https://www.nature.com/articles/s41588-022-01148-2) | This task uses variants from the Deciphering Developmental Disorders (DDD) and autism spectrum disorder (ASD) cohort study, which catalogs genetic mutations linked to rare pediatric and developmental diseases, to evaluate classification of pathogenic versus benign variants based on coding sequence context.|
| [mRNA Translation Efficiency](https://www.biorxiv.org/content/10.1101/2024.08.11.607362v2.full.pdf) | This task predicts ribosome profiling signal intensity along coding sequences, evaluating how well models capture translation efficiency and codon-level regulation from sequence context.|
| [Protein Abundance](https://www.biorxiv.org/content/10.1101/2023.09.09.556981v1.full.pdf) | This task predicts fluorescent protein expression levels (mRFP) from coding sequences, testing how accurately models capture codon-dependent effects on translation efficiency and protein abundance. |

**Data Collection Method for all data Dataset:**
* Human <br>

**Labeling Method by Dataset:**
* Not Applicable <br>

# Inference:
**Acceleration Engine:** None <br>
**Test Hardware:** A100 <br>

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

For more detailed information on ethical considerations for this model, please see the Model Card++ Bias, Explainability, Safety & Security, and Privacy Subcards.

Users are responsible for ensuring the physical properties of model-generated molecules are appropriately evaluated and comply with applicable safety regulations and ethical standards. <br>

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).  <br>
