---

# Explainable Hate Speech Detection Using Generative AI

![Project Architecture](https://i.imgur.com/T3O9dch.png)

This project explores the detection of **hate speech** and **fake news** on online platforms using **Quantized Low Rank Adaptation (QLoRA)** and **Retrieval-Augmented Generation (RAG)**. It utilizes state-of-the-art NLP techniques to address the challenge of identifying harmful content with speed and accuracy.

---

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Screenshots](#screenshots)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Scope](#future-scope)
- [Contributors](#contributors)

---

## Introduction

Online platforms are overwhelmed with harmful content, including **hate speech** and **fake news**, which are difficult to moderate manually. This project aims to leverage **Generative AI**, specifically **QLoRA** and **RAG**, to efficiently detect and classify these types of content. By combining **retrieval-based** and **generation-based** models, our approach enables quick and accurate detection, ensuring safer digital environments.

---

## Features

- **Transliteration and Translation**: Utilizes **IndicXlit** and **IndicTrans2** to transliterate and translate code-mixed Hindi-English text into standard Hindi and English for processing.
- **Embedding Extraction**: Embeddings are extracted using advanced language models such as **BERT**, **XLM-Roberta**, **Electra**, and others.
- **QLoRA Fine-Tuning**: Efficient fine-tuning of large language models with **QLoRA**, reducing memory requirements without compromising performance.
- **RAG (Retrieval-Augmented Generation)**: Enhances model performance by retrieving relevant documents from a vector database and generating well-informed responses based on retrieved data.

---

## Model Architecture

### Workflow Overview

1. **Data Preprocessing**:
   - Text is cleaned, transliterated, and translated into English using **IndicXlit** and **IndicTrans2**.
   - Special characters and unwanted symbols are removed.
   - The dataset is manually labeled as **Hate/Non-Hate** and **Fake/Non-Fake**.

2. **QLoRA Fine-Tuning**:
   - LoRA fine-tunes large language models efficiently by adjusting only specific parameters using low-rank matrix decomposition.
   - **QLoRA** adds quantization to this process, reducing computational costs while maintaining accuracy.

   ![LoRA Architecture](https://i.imgur.com/uICHyfA.png)

3. **Retrieval-Augmented Generation (RAG)**:
   - **RAG** fetches relevant documents from a vector database using **LangChain** and integrates them into the model's generation process.
   - This method ensures that the language model has access to up-to-date information, improving response quality.

   ![RAG Architecture](https://i.imgur.com/GNhC4pq.png)

4. **Hyper parameters used for Training 7B’s**:
   - The following hyperparameters were used to fine-tune the 7B models (such as Mistral 7B, DeepSeek 7B, Zephyr Alpha 7B, and Zephyr Beta 7B) with QLoRA for the task of hate speech and fake news detection:

   ![RAG Architecture](https://i.imgur.com/1PGB4yJ.png)

   These hyperparameters were selected to balance performance while minimizing computational overhead during the fine-tuning process. By adjusting parameters like LoRA rank and dropout, we optimized the model's performance without overwhelming GPU resources.

---

## Screenshots

### Data Processing and Architecture

<table>
  <tr>
    <td align="center">
      <strong>LoRA Architecture</strong><br>
      <img src="https://i.imgur.com/uICHyfA.png" alt="LoRA Architecture" width="400"/>
    </td>
    <td align="center">
      <strong>RAG Workflow</strong><br>
      <img src="https://i.imgur.com/GNhC4pq.png" alt="RAG Workflow" width="400"/>
    </td>
  </tr>
</table>

### Results Comparison

<table>
  <tr>
    <td align="center">
      <strong>Hate Detection Results</strong><br>
      <img src="https://i.imgur.com/Jix6AnA.png" alt="Hate Detection Results" width="400"/>
    </td>
    <td align="center">
      <strong>Fake Detection Results</strong><br>
      <img src="https://i.imgur.com/K40HNoU.png" alt="Fake Detection Results" width="400"/>
    </td>
  </tr>
</table>

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your_username/explainable-hate-speech-detection.git
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up the environment**:
   - Add API keys and model paths in the configuration file.

---

## Usage

To run the model:

1. **Preprocess the Data**:
   ```bash
   python preprocess.py --input_path path_to_data
   ```

2. **Train the Model**:
   ```bash
   python train.py --model QLoRA
   ```

3. **Evaluate the Model**:
   ```bash
   python evaluate.py --test_data path_to_test_data
   ```

---

## Results

| Model Name      | Hate Macro F1 with QLoRA | Hate Macro F1 with RAG & QLoRA |
|-----------------|--------------------------|--------------------------------|
| Mistral 7B      | 72.3                      | 72.8                          |
| DeepSeek 7B     | 72.3                      | 70.9                          |
| Zephyr Beta 7B  | 69.6                      | 70.8                          |
| Zephyr Alpha 7B | 67.1                      | 69.7                          |

| Model Name      | Fake Macro F1 with QLoRA | Fake Macro F1 with RAG & QLoRA |
|-----------------|--------------------------|--------------------------------|
| Mistral 7B      | 77.3                      | 78.2                          |
| DeepSeek 7B     | 74.7                      | 78.4                          |
| Zephyr Beta 7B  | 77.3                      | 78.2                          |
| Zephyr Alpha 7B | 74.7                      | 78.4                          |

---

## Future Scope

- **Text Generation**: Future iterations of this model will incorporate the generation of explanatory text, providing reasons for the classification of content as **Hate/Non-Hate** and **Fake/Non-Fake**.
- **Enhanced LLMs**: We plan to fine-tune larger language models for better performance in text classification and reasoning tasks.
- **Expanded Dataset**: Gathering more diverse and extensive datasets for improved accuracy.

---
