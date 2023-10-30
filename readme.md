## Code Version
- PyTorch

## Pretrained Model
- [Chinese BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)

## Data and Code
- [CMRE Repository](https://github.com/cnunlp/CMRE)

## 1. Code Structure

```plaintext
.
├── src  
│   ├── metaphor.py  
│   ├── demo.py  
│   ├── metrics.py  
│   ├── utils.py  
│   ├── experiments.conf  
│   └── requirements.txt  
├── bert  
│   ├── modeling.py  
│   ├── optimization.py  
│   └── tokenization.py  
├── data  
│   ├── dev  
│   ├── test  
│   └── train  
└── pretrain_model  
    ├── config.json  
    ├── pytorch_model.bin  
    └── vocab.txt  

----


**structure explaination**：


**metaphor.py**：metaphor identification model script

**demo.py**：training & testing of metaphor identification  script

**metrics.py**：the ways to evaluate dev data

**utils.py**：data conversion and reading script

**experiments.conf**：parameter configuration file required for code running 

**requirements.txt**：the necessary environment files for the code to run

**bert**：used to store bert model related script files 

**data**：used to store the training validation prediction file and the final prediction result file

**pretrained_model**：configuration file, vocab.txt, model



## Running Environment

- **Requirements**: 
  - Python version **3.7** and above.
  - Refer to `requirements.txt` for environment configuration.

## Running the Model

### 1. Selecting the Model

Choose from the following options:
- **FULL**
- **STF-None**
- **ITC**

### 2. Adjusting Configuration File (`experiments.conf`)

Modify the `experiments.conf` based on your model choice:

- **FULL** or **ITC**: Ensure the `use_span_type` parameter is set to `true`.
- **STF-None**: Crucially set the `use_span_type` parameter to `false`.

### 3. Executing the Command

Execute the model with one of these commands:

- Directly with Python: 
  - `python demo.py`
  - `python demo_ITC.py`
  
- Using the shell script:
  - `sh test.sh`
  - **Note**: Specify the model name as an argument. 
    - Example: `sh test.sh FULL` 

> **Tip**: Double-check configurations in `experiments.conf` before running to ensure accurate results.
