# Chinese Metaphorical Relation Extraction: Dataset and Models
> **EMNLP 2023-findings**

![Model Version](https://img.shields.io/badge/Model-PyTorch-blue) ![Paper](https://img.shields.io/badge/Paper-EMNLP2023-green)

**Authors:** Guiha Chen, Tiantian Wu, Miaomiao Cheng, Xu Han, Jiefu Gong, Shijin Wang, and Wei Song.

---

## 📋 Table of Contents
- [Introduction](#introduction)
- [Code Structure](#code-structure)
- [Environment and Requirements](#environment-and-requirements)
- [Running the Model](#running-the-model)
- [Resources](#resources)


---

## 📌 Introduction

Metaphor identification has traditionally been approached as a sequence labeling or a syntactically related word-pair classification problem. This research introduces a novel formulation, viewing metaphor identification as a relation extraction problem. The paper proposes metaphorical relations, which are connections between two spans in sentences: a target span and a source-related span. This approach allows for more flexible and precise text units beyond single words, capturing the properties of the target and the source. The research also introduces a dataset for Chinese metaphorical relation extraction, consisting of over 4,200 sentences annotated with metaphorical relations, target/source-related spans, and fine-grained span types. The dataset and models aim to bridge the gap between linguistic and conceptual metaphor processing.

[📜 Read the full paper](https://myaidrive.com/6ABGmy6hsiYLDFTY/2312_chinese.pdf)

---

## 📂 Code Structure

```plaintext
.
├── src/
│   ├── metaphor.py
│   ├── demo.py
│   ├── metrics.py
│   ├── utils.py
│   ├── experiments.conf
│   └── requirements.txt
├── bert/
│   ├── modeling.py
│   ├── optimization.py
│   └── tokenization.py
├── data/
│   ├── dev/
│   ├── test/
│   └── train/
└── pretrain_model/
    ├── config.json
    ├── pytorch_model.bin
    └── vocab.txt

  
---

## 🛠 Environment and Requirements

- **Python version:** 3.8 or above.
- **Dependencies:** Refer to `requirements.txt` for the complete list.

---

## 🚀 Running the Model

1. **Select the Model:** 
   - FULL
   - STF-None
   - ITC

2. **Adjust Configuration File (`experiments.conf`):** 
   - For **FULL** or **ITC**: Set `use_span_type` to `true`.
   - For **STF-None**: Set `use_span_type` to `false`.

3. **Execute the Model:** 
   - Using Python: 
     - `python demo.py`
     - `python demo_ITC.py`
   - Using the shell script:
     - `sh test.sh`
     - Example: `sh test.sh FULL`

> **Tip**: Ensure configurations in `experiments.conf` are correct before execution.

---

## 📚 Resources

- **Pretrained Model:** [Chinese BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)
- **Dataset and Code:** [CMRE Repository](https://github.com/cnunlp/CMRE)

