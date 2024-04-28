# Chinese Metaphorical Relation Extraction
> **EMNLP 2023-findings**

![Model Version](https://img.shields.io/badge/Model-PyTorch-blue) ![Paper](https://img.shields.io/badge/Paper-EMNLP2023-green)

**Authors:** Guiha Chen, Tiantian Wu, Miaomiao Cheng, Xu Han, Jiefu Gong, Shijin Wang, and Wei Song.

---

## 📋 Table of Contents
- [Introduction](#anchor-introduction)
- [Code Structure](#anchor-code-structure)
- [Environment and Requirements](#anchor-environment-and-requirements)
- [Running the Model](#anchor-running-the-model)
- [Resources](#anchor-resources)



---

<a id="anchor-introduction"></a>
## 📌 Introduction

Metaphor identification has traditionally been approached as a sequence labeling or a syntactically related word-pair classification problem. This research introduces a novel formulation, viewing metaphor identification as a relation extraction problem. The paper proposes metaphorical relations, which are connections between two spans in sentences: a target span and a source-related span. This approach allows for more flexible and precise text units beyond single words, capturing the properties of the target and the source. The research also introduces a dataset for Chinese metaphorical relation extraction, consisting of over 4,200 sentences annotated with metaphorical relations, target/source-related spans, and fine-grained span types. The dataset and models aim to bridge the gap between linguistic and conceptual metaphor processing.

[📜 Read the full paper](https://openreview.net/forum?id=RO460OVpev&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DEMNLP%2F2023%2FConference%2FAuthors%23your-submissions))

---

<a id="anchor-code-structure"></a>
## 📂 Code Structure


.  
├── 📁 src  
│   ├── 📄 metaphor.py  
│   ├── 📄 demo.py  
│   ├── 📄 metrics.py  
│   ├── 📄 utils.py  
│   ├── 📄 experiments.conf  
│   └── 📄 requirements.txt  
├── 📁 data  
│   ├── 📄 dev_data.jsonlines  
│   ├── 📄 test_data.jsonlines  
│   └── 📄 train_data.jsonlines  
└── 📁 pretrain_model  
    ├── 📄 config.json  
    ├── 📄 pytorch_model.bin  
    └── 📄 vocab.txt  


---


<a id="anchor-environment-and-requirements"></a>
## 🛠 Environment and Requirements

- **Python version:** 3.8 or above.
- **Dependencies:** Refer to `requirements.txt` for the complete list.

---

<a id="anchor-running-the-model"></a>
## 🚀 Running the Model

## 1. Select the Model:

### FULL
- **Description:** This is the comprehensive model proposed in the paper. It adopts a pair-wise span type classification manner and enhances the interaction between span type classification and relation extraction. The FULL model views span type classification as a pair-wise task, where the types of two spans are predicted simultaneously since they are closely associated. This approach is crucial for capturing the meaning and structure of metaphors. When using the FULL model, ensure that the `use_span_type` is set to `true` in the configuration.
   
### STF-None
- **Description:** The STF-None model is similar to the FULL model, but it does not utilize span type features during metaphorical relation extraction. Ablation studies in the paper showed that when span type features are removed, there's a decrease in F1 performance for both metaphorical relation extraction and span extraction. For this model, set the `use_span_type` to `false` in the configuration.
   
### ITC (Independent Span Type Classification)
- **Description:** The ITC model replaces the pair-wise span classification module with an independent span type classification module. This means that each span's type is predicted independently, rather than in pairs. However, during metaphorical relation extraction, the span type features are still utilized. The paper indicates that when switching to an independent span type classification, there's a decrease in F1 performance for metaphorical relation extraction and span extraction. When using the ITC model, ensure that the `use_span_type` is set to `true` in the configuration.

## 2. Adjust Configuration File (`experiments.conf`): 
- For **FULL** or **ITC**: Set `use_span_type` to `true`.
- For **STF-None**: Set `use_span_type` to `false`.

## 3. Execute the Model: 
- Using Python: 
  - `python demo.py`
  - `python demo_ITC.py`
- Using the shell script:
  - `sh test.sh`
  - Example: `sh test.sh FULL`

> **Tip**: Ensure configurations in `experiments.conf` are correct before execution.


## 📚 Resources

- **Pretrained Model:** [Chinese-BERT-WWM-ext](https://huggingface.co/hfl/chinese-bert-wwm-ext)
- **BERT Source Code:** [BERT GitHub Repository](https://github.com/google-research/bert)
- **Dataset and Code:** [CMRE Repository](https://github.com/cnunlp/CMRE)
- **FULL_model:** [Baidu Netdisk]链接：https://pan.baidu.com/s/1i8-XBClgFYbwJGIQfTk-sg  提取码：oeln 
  - **Note:** The best seed model for our FULL_model can be found in the provided link.


