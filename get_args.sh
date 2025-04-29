#!/bin/bash
# shellcheck disable=SC2034
# shellcheck disable=SC2181

ARGS=$(getopt -o t: --long dataset_name:,dataset_root_path:,tr:,ars:,an:,max_sample_num_per_user:,max_ub_num_quantile:,max_ui_num_quantile:,max_bi_num_quantile:,lightgcn_layers:,noise_schedule:,noise_scale:,min_noise:,max_noise:,max_diffusion_steps:,embedding_dim:,device:,train_batch_size:,augment_batch_size:,lr:,epochs:,eval_interval:,early_stop:,save_model:,seed:,output_root_path:,config_path:,prefix:,is_augment: -n "$0" -- "$@")

if [ $? != 0 ]; then
  echo "parse args error"
  exit 1
fi

eval set -- "$ARGS"

while true
do
  case "$1" in
    -d|--dataset_name)
      dataset_name=$2
      shift 2
      ;;
    -r|--dataset_root_path)
      dataset_root_path=$2
      shift 2
      ;;
    -a|--tr)
      tr=$2
      shift 2
      ;;
    -n|--ars)
      ars=$2
      shift 2
      ;;
    -l|--an)
      an=$2
      shift 2
      ;;
    --max_sample_num_per_user)
      max_sample_num_per_user=$2
      shift 2
      ;;
    --max_ub_num_quantile)
      max_ub_num_quantile=$2
      shift 2
      ;;
    --max_ui_num_quantile)
      max_ui_num_quantile=$2
      shift 2
      ;;
    --max_bi_num_quantile)
      max_bi_num_quantile=$2
      shift 2
      ;;
    -s|--lightgcn_layers)
      lightgcn_layers=$2
      shift 2
      ;;
    -c|--noise_schedule)
      noise_schedule=$2
      shift 2
      ;;
    -e|--noise_scale)
      noise_scale=$2
      shift 2
      ;;
    -i|--min_noise)
      min_noise=$2
      shift 2
      ;;
    -t|--max_noise)
      max_noise=$2
      shift 2
      ;;
    -b|--max_diffusion_steps)
      max_diffusion_steps=$2
      shift 2
      ;;
    -u|--embedding_dim)
      embedding_dim=$2
      shift 2
      ;;
    --device)
      device=$2
      shift 2
      ;;
    --train_batch_size)
      train_batch_size=$2
      shift 2
      ;;
    --augment_batch_size)
      augment_batch_size=$2
      shift 2
      ;;
    --lr)
      lr=$2
      shift 2
      ;;
    --epochs)
      epochs=$2
      shift 2
      ;;
    --eval_interval)
      eval_interval=$2
      shift 2
      ;;
    --early_stop)
      early_stop=$2
      shift 2
      ;;
    --save_model)
      save_model=$2
      shift 2
      ;;
    --seed)
      seed=$2
      shift 2
      ;;
    --output_root_path)
      output_root_path=$2
      shift 2
      ;;
    --config_path)
      config_path=$2
      shift 2
      ;;
    --prefix)
      prefix=$2
      shift 2
      ;;
    --is_augment)
      is_augment=$2
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "parse args error"
      exit 1
      ;;
  esac
done

# default args
if [ -z "$dataset_name" ]; then
  dataset_name='Youshu'
fi

if [ -z "$dataset_root_path" ]; then
  dataset_root_path='./raw_data'
fi

if [ -z "$tr" ]; then
  tr='(6,70)'
fi

if [ -z "$ars" ]; then
  ars='[(3,4),(5,5)]'
fi

if [ -z "$an" ]; then
  an=1
fi

if [ -z "$max_sample_num_per_user" ]; then
  max_sample_num_per_user=3
fi

if [ -z "$max_ub_num_quantile" ]; then
  max_ub_num_quantile=0.95
fi

if [ -z "$max_ui_num_quantile" ]; then
  max_ui_num_quantile=0.95
fi

if [ -z "$max_bi_num_quantile" ]; then
  max_bi_num_quantile=0.95
fi

if [ -z "$lightgcn_layers" ]; then
  lightgcn_layers=10
fi

if [ -z "$noise_schedule" ]; then
  noise_schedule='linear'
fi

if [ -z "$noise_scale" ]; then
  noise_scale=0.1
fi

if [ -z "$min_noise" ]; then
  min_noise=0.1
fi

if [ -z "$max_noise" ]; then
  max_noise=1.0
fi

if [ -z "$max_diffusion_steps" ]; then
  max_diffusion_steps=20
fi

if [ -z "$embedding_dim" ]; then
  embedding_dim=128
fi

if [ -z "$device" ]; then
  device='cuda'
fi

if [ -z "$train_batch_size" ]; then
  train_batch_size=512
fi

if [ -z "$augment_batch_size" ]; then
  augment_batch_size=512
fi

if [ -z "$lr" ]; then
  lr=0.001
fi

if [ -z "$epochs" ]; then
  epochs=3000
fi

if [ -z "$eval_interval" ]; then
  eval_interval=10
fi

if [ -z "$early_stop" ]; then
  early_stop=20
fi

if [ -z "$save_model" ]; then
  save_model=True
fi

if [ -z "$seed" ]; then
  seed=2024
fi

if [ -z "$output_root_path" ]; then
  output_root_path='./output'
fi

if [ -z "$prefix" ]; then
  prefix='user_bundle_train'
fi

if [ -z "$is_augment" ]; then
  is_augment='True'
fi
