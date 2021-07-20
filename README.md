# RTGCN
Relational Temporal Graph Convolutional Network for Ranking-based Stock Prediction

Data and pretained model have been uploaded to [Google drive](https://drive.google.com/drive/folders/1UaQM_KLf7hG2IJUN-niqICMwfr3jb_Ci?usp=sharing)

Place the data on the path: RTGCN/data/

Place the pretained model on the path: RTGCN/work_dir/NASDAQ(OR NYSE)

Tesing pretained model on NASDAQ. Edit the "weights" and "relation_path" variables to reload the other models weights.
--------
    python main.py recognition -c config/rt_gcn/NASDAQ/test.yaml
Testing pretained model on NYSE
--------
    python main.py recognition -c config/rt_gcn/NYSE/test.yaml
