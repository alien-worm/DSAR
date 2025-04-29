import os
import time
import random
import numpy as np

import argparse
import configparser

import torch

from utils.logger import Logger

from datasets.bundle_seq_dataset import BundleSeqDataProcessor
from bundle_seq_diffusion_model_trainer import BundleSeqDiffusionModelTrainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True


def read_args_from_config(commandline_args: argparse.Namespace) -> argparse.Namespace:
    config = configparser.ConfigParser()
    config.read(commandline_args.config_path)

    commandline_args = vars(commandline_args)
    for section in config.sections():
        for key, value in config.items(section):
            if key in commandline_args and commandline_args[key] is None:
                commandline_args[key] = eval(value)

    return argparse.Namespace(**commandline_args)


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Bundle Sequence Diffusion Model')
    # Dataset arguments
    parser.add_argument('--dataset_name', type=str, help='dataset name: NetEase, Youshu')
    parser.add_argument('--dataset_root_path', type=str, help='dataset root path')
    parser.add_argument('--tr', type=str, help='train bundle interaction num range')
    parser.add_argument('--ars', type=str, help='a list of augment bundle interaction num range')
    parser.add_argument('--an', type=int, help='augment num')
    parser.add_argument('--max_sample_num_per_user', type=int, help='max sample num per user')
    parser.add_argument('--max_ub_num_quantile', type=float, help='max user bundle interaction num quantile')
    parser.add_argument('--max_ui_num_quantile', type=float, help='max user item interaction num quantile')
    parser.add_argument('--max_bi_num_quantile', type=float, help='max bundle item interaction num quantile')

    # Model arguments
    parser.add_argument('--lightgcn_layers', type=int, help='lightgcn layers')
    parser.add_argument('--noise_schedule', type=str, help='noise schedule')
    parser.add_argument('--noise_scale', type=float, help='noise scale')
    parser.add_argument('--min_noise', type=float, help='min noise')
    parser.add_argument('--max_noise', type=float, help='max noise')
    parser.add_argument('--max_diffusion_steps', type=int, help='max diffusion steps')
    parser.add_argument('--embedding_dim', type=int, help='embedding dim')

    # Training arguments
    parser.add_argument('--device', type=str, help='device')
    parser.add_argument('--train_batch_size', type=int, help='train batch size')
    parser.add_argument('--augment_batch_size', type=int, help='agument batch size')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--epochs', type=int, help='epochs')
    parser.add_argument('--eval_interval', type=int, help='eval interval')
    parser.add_argument('--early_stop', type=int, help='early stop')
    parser.add_argument('--save_model', type=bool, help='save model')
    parser.add_argument('--seed', type=int, help='random seed')

    # output arguments
    parser.add_argument('--output_root_path', type=str, help='output root path')
    parser.add_argument('--config_path', type=str, help='config path')

    commandline_args = parser.parse_args()

    # combing config file and commandline arguments.
    # Commandline arguments have higher priority!!!
    if os.path.exists(commandline_args.config_path):
        commandline_args = read_args_from_config(commandline_args)

    if type(commandline_args.tr) == str:
        commandline_args.tr = eval(commandline_args.tr)

    if type(commandline_args.ars) == str:
        commandline_args.ars = eval(commandline_args.ars)

    return commandline_args


def args_to_prefix(args: argparse.Namespace) -> str:
    prefix = 'Data-{}-{}-{}-{}-{}_Model-{}-{}-{}-{}-{}-{}-{}'.format(
        args.dataset_name,
        args.tr,
        args.ars,
        args.an,
        args.max_sample_num_per_user,
        args.lightgcn_layers,
        args.noise_schedule,
        args.noise_scale,
        args.min_noise,
        args.max_noise,
        args.max_diffusion_steps,
        args.embedding_dim
    )
    prefix = prefix.replace(' ', '')
    return prefix


def main():
    args = args_parser()
    prefix = args_to_prefix(args)
    set_seed(args.seed)

    output_path = os.path.join(
        args.output_root_path,
        prefix + '_' + time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    )

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    logger = Logger(root_path=output_path)
    logger.divider('Arguments')
    logger.info(args)
    logger.divider('Arguments', end=True)

    bundle_seq_data_processor = BundleSeqDataProcessor(
        dataset_name=args.dataset_name,
        dataset_root_path=args.dataset_root_path,
        train_bundle_interaction_num_range=args.tr,
        augment_bundle_interaction_num_range_list=args.ars,
        max_sample_num_per_user=args.max_sample_num_per_user,
        max_user_bundle_interaction_num_quantile=args.max_ub_num_quantile,
        max_user_item_interaction_num_quantile=args.max_ui_num_quantile,
        max_bundle_item_interaction_num_quantile=args.max_bi_num_quantile,
        augment_num=args.an,
        device=args.device,
        logger=logger
    )

    bundle_seq_train_dataset = bundle_seq_data_processor.get_train_dataset()
    bundle_seq_train_dataloader = torch.utils.data.DataLoader(
        bundle_seq_train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True
    )

    bundle_seq_diffusion_model_trainer = BundleSeqDiffusionModelTrainer(
        prefix=prefix,
        output_path=output_path,
        bundle_seq_data_processor=bundle_seq_data_processor,
        bundle_seq_train_dataloader=bundle_seq_train_dataloader,
        lightgcn_layers=args.lightgcn_layers,
        noise_schedule=args.noise_schedule,
        noise_scale=args.noise_scale,
        min_noise=args.min_noise,
        max_noise=args.max_noise,
        max_diffusion_steps=args.max_diffusion_steps,
        embedding_dim=args.embedding_dim,
        device=args.device,
        logger=logger
    )

    bundle_seq_diffusion_model_trainer.train(
        epochs=args.epochs,
        learning_rate=args.lr,
        eval_interval=args.eval_interval,
        early_stop=args.early_stop
    )

    # if args.save_model:
    #     bundle_seq_diffusion_model_trainer.save_model()

    augment_dataset_dataloader_list = []
    for augment_bundle_interaction_num_range in args.ars:
        bundle_seq_augment_dataset = bundle_seq_data_processor.get_augment_dataset(
            augment_bundle_interaction_num_range=augment_bundle_interaction_num_range
        )
        augment_dataset_dataloader = torch.utils.data.DataLoader(
            bundle_seq_augment_dataset,
            batch_size=args.augment_batch_size,
            shuffle=False
        )
        augment_dataset_dataloader_list.append(augment_dataset_dataloader)

    bundle_seq_diffusion_model_trainer.augmentation(
        augment_bundle_interaction_num_range_list=args.ars,
        augment_dataset_dataloader_list=augment_dataset_dataloader_list
    )

    logger.close()


if __name__ == '__main__':
    main()
