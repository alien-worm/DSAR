# DSAR
## 1. Requirements
- Python 3.7
- torch 1.13.1+cu171
- torch_geometric 2.2.0
- numpy 1.21.6
- pandas 1.1.5
- sqlite3 2.6.0

## 2. Datasets
- raw_data/Youshu
- raw_data/NetEase

## 3. Scripts
### 3.1 Scripts files
- get_args.py: parse arguments(with default values) from command line
- run.sh: run the model with specified arguments
### 3.2 Parameters
- dataset_name: the name of the dataset
- dataset_root_path: the root path of the dataset
- tr: the tuple of the training ratio and the testing ratio
- ars: the list of the tuple of the augment ratio and the sample ratio
- an: the number of the augment networks
- max_sample_num_per_user: the maximum number of the samples per user
- max_ub_num_quantile: the quantile of the maximum number of the users per batch
- max_ui_num_quantile: the quantile of the maximum number of the items per batch
- max_bi_num_quantile: the quantile of the maximum number of the batches per item
- lightgcn_layers: the number of the layers of the LightGCN
- noise_schedule: the schedule of the noise
- noise_scale: the scale of the noise
- max_noise: the maximum noise
- min_noise: the minimum noise
- max_diffusion_steps: the maximum number of the diffusion steps
- embedding_dim: the dimension of the embedding
- device: the device of the model
- train_batch_size: the batch size of the training
- augment_batch_size: the batch size of the augment
- lr: the learning rate
- epochs: the number of the epochs
- eval_interval: the interval of the evaluation
- early_stop: the early stop
- save_model: whether to save the model
- seed: the seed of the random number generator
- output_root_path: the root path of the output
- config_path: the path of the config file

## 4. Example of running the model on three datasets
### 4.1 Youshu
```shell
./run.sh --dataset_name Youshu --tr "(5,70)" --ars "[(4,4)]" --an 1 --max_sample_num_per_user 1 --max_ub_num_quantile 1.0 --max_ui_num_quantile 1.0 --max_bi_num_quantile 1.0
```

### 4.2 NetEase
```shell
./run.sh --dataset_name NetEase --tr "(5,6)" --ars "[(4,4)]" --an 1 --max_sample_num_per_user 1 --augment_batch_size 256 --train_batch_size 256 --max_ub_num_quantile 0.99 --max_ui_num_quantile 0.99 --max_bi_num_quantile 0.99
```