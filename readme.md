-code version: PyTorch

-pretrained model:https://github.com/ymcui/Chinese-BERT-wwm

-data and code : https://github.com/cnunlp/CMRE.



1.code structure

.  
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



2. Running environment

- The operating environment requires the python version to be 3.7 and above. For the operating environment configuration, see `requirements.txt`


 
3. Running the Model

①. Selecting the Model:
Choose the model you wish to run from the following options:
- FULL
- STF-None
- ITC

②. Adjusting Configuration File (experiments.conf):
Depending on the chosen model, make sure to modify the experiments.conf file accordingly:
- For the FULL or ITC models: Ensure the use_span_type parameter is set to true.
- For the STF-None model: It's crucial to set the use_span_type parameter to false.

③. Executing the Command:
After configuring the parameters based on your model choice, execute the model using one of the following commands:
- To run directly with Python: python demo.py or python demo_ITC.py
- If using the shell script, execute: sh test.sh. Remember to specify the model name as an argument, e.g., sh test.sh FULL, sh test.sh STF-None, or sh test.sh ITC.

Please double-check the configurations in experiments.conf before running to ensure accurate and expected results.

