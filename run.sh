#!/bin/bash -e
# shellcheck disable=SC2154

source get_args.sh

# if is_augment is True, run user-bundle data augmentation
if [ "$is_augment" = "True" ]; then
  # user-bundle data augmentation
  python main.py \
    --dataset_name "$dataset_name" \
    --dataset_root_path "$dataset_root_path" \
    --tr "$tr" \
    --ars "$ars" \
    --an "$an" \
    --max_sample_num_per_user "$max_sample_num_per_user" \
    --max_ub_num_quantile "$max_ub_num_quantile" \
    --max_ui_num_quantile "$max_ui_num_quantile" \
    --max_bi_num_quantile "$max_bi_num_quantile" \
    --lightgcn_layers "$lightgcn_layers" \
    --noise_schedule "$noise_schedule" \
    --noise_scale "$noise_scale" \
    --min_noise "$min_noise" \
    --max_noise "$max_noise" \
    --max_diffusion_steps "$max_diffusion_steps" \
    --embedding_dim "$embedding_dim" \
    --device "$device" \
    --train_batch_size "$train_batch_size" \
    --augment_batch_size "$augment_batch_size" \
    --lr "$lr" \
    --epochs "$epochs" \
    --eval_interval "$eval_interval" \
    --early_stop "$early_stop" \
    --save_model "$save_model" \
    --seed "$seed" \
    --output_root_path "$output_root_path" \
    --config_path "$config_path"

  prefix="Data-${dataset_name}-${tr}-${ars}-${an}-${max_sample_num_per_user}_Model-${lightgcn_layers}-${noise_schedule}-${noise_scale}-${min_noise}-${max_noise}-${max_diffusion_steps}-${embedding_dim}"
fi

cd ./baselines

cd ./CrossCBR
python train.py -g 0 -m CrossCBR -d "$dataset_name" --prefix "$prefix" --info "$prefix"

cd ../MultiCBR
python train.py -g 0 -m MultiCBR -d "$dataset_name" --prefix "$prefix" --info "$prefix"

cd ../BGCN
python main.py -d "$dataset_name" --prefix "$prefix"

cd ../PET
python train.py -g 0 -d "$dataset_name" --prefix "$prefix"

cd ../CoHeat
python main.py --data "$dataset_name" --prefix "$prefix"

cd ../UHBR
python main.py --dataset "$dataset_name" --prefix "$prefix"