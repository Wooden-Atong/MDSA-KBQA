# MDSA-KBQA
Source code and data for *Multi-hop Question Answering System Based on Multi-dimensional Information Alignment* .

---

## Dependencies
 - Python>=3.7
 - Pytorch<=1.9
 - nltk
 - numpy
 - pandas
 - scikit_learn
 - scipy
 - tqdm


## Prepare Data
Before running the models, some preprocessing is needed:
 - Download preprocessed datasets from [google drive](https://drive.google.com/drive/folders/1qRXeuoL-ArQY7pJFnMpNnBu0G-cOz6xv?usp=sharing) into `data/`.
 - Unzip `data/CWQ.tgz` and `data/webqsp.tgz`
 
(Optional) If you want to start from the raw data, you can find instructions to obtain datasets used in this repo in [preprocessing](https://github.com/RichardHGL/WSDM2021_NSM/tree/main/preprocessing) folder


## Run MDSA-KBQA
```
bash train_CWQ.sh
bash train_webqsp.sh
```

Important arguments:

 --data_folder          Path to load dataset.
 --checkpoint_dir       Path to save checkpoint and logs.
 --num_step             Multi-hop reasoning steps, hyperparameters.
 --entity_dim           Hidden size of reasoning module.
 --eval_every           Number of interval epoches between evaluation.
 --experiment_name      The name of log and ckpt. If not defined, it will be generated with timestamp.
 --eps                  Accumulated probability to collect answers, used to generate answers and affect Precision, Recalll and F1 metric.
 --use_self_loop        If set, add a self-loop edge to all graph nodes.
 --use_inverse_relation If set, add reverse edges to graph.
 --encode_type          If set, use type layer initialize entity embeddings. 
 --load_experiment      Path to load trained ckpt, only relative path to --checkpoint_dir is acceptable. 
 --is_eval              If set, code will run fast evaluation mode on test set with trained ckpt from --load_experiment option.
 --reason_kb            If set, model will reason step by step. Otherwise, model may focus on all nodes on graph every step.
 --load_teacher         Path to load teacher ckpt, only relative path to --checkpoint_dir is acceptable. 
 
For the WebQSP dataset, the `num_step` is selected from {3, 5}, and for the CWQ dataset, it is chosen from {4, 7}.


## Citation
Please cite our paper if this repository inspires your work.
```

```

## Contact
If you have any questions regarding the code, please create an issue or contact the owner of this repository.