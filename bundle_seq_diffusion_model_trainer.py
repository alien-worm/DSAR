import os
import time
import pandas as pd

import torch
import torch_geometric
from transformers import BertConfig

from datasets.bundle_seq_dataset import BundleSeqDataProcessor
from models.bundle_seq_diffusion_model import BundleSeqDiffusionModel
from utils.logger import Logger


class BundleSeqDiffusionModelTrainer(object):

    def __init__(
            self,
            prefix: str,
            output_path: str,
            bundle_seq_data_processor: BundleSeqDataProcessor,
            bundle_seq_train_dataloader: torch.utils.data.DataLoader,
            lightgcn_layers: int = 3,
            noise_schedule: str = 'linear',
            noise_scale: float = 0.1,
            min_noise: float = 0.1,
            max_noise: float = 1,
            max_diffusion_steps: int = 20,
            embedding_dim: int = 128,
            bert_config: BertConfig = None,
            device: str = 'cpu',
            logger: Logger = None,
    ):
        self.params = {
            'user_num': bundle_seq_data_processor.user_num,
            'bundle_num': bundle_seq_data_processor.bundle_num,
            'item_num': bundle_seq_data_processor.item_num,
            'train_bundle_interaction_num_range': bundle_seq_data_processor.train_bundle_interaction_num_range,
            'augment_num': bundle_seq_data_processor.augment_num,
            'lightgcn_layers': lightgcn_layers,
            'noise_schedule': noise_schedule,
            'noise_scale': noise_scale,
            'min_noise': min_noise,
            'max_noise': max_noise,
            'max_diffusion_steps': max_diffusion_steps,
            'embedding_dim': embedding_dim,
            'bert_config': bert_config,
            'device': device
        }
        self.prefix = prefix

        self.output_path = output_path
        self.bundle_seq_data_processor = bundle_seq_data_processor
        self.bundle_seq_train_dataloader = bundle_seq_train_dataloader

        self.logger = logger if logger is not None else Logger(root_path=output_path, file_name=self.prefix)

        self.bundle_seq_diffusion_model = self._init_bundle_seq_diffusion_model()

    def _init_bundle_seq_diffusion_model(self) -> BundleSeqDiffusionModel:
        start_time = time.time()
        self.logger.divider('Bundle Diffusion Model Initialization')
        self.logger.info('Parameters: {}'.format(self.params))

        bundle_seq_diffusion_model = BundleSeqDiffusionModel(
            user_num=self.params['user_num'],
            bundle_num=self.params['bundle_num'],
            item_num=self.params['item_num'],
            augment_num=self.params['augment_num'],
            lightgcn_layers=self.params['lightgcn_layers'],
            noise_schedule=self.params['noise_schedule'],
            noise_scale=self.params['noise_scale'],
            min_noise=self.params['min_noise'],
            max_noise=self.params['max_noise'],
            max_diffusion_steps=self.params['max_diffusion_steps'],
            embedding_dim=self.params['embedding_dim'],
            bert_config=self.params['bert_config'],
            device=self.params['device']
        ).to(self.params['device'])

        self.logger.info('Bundle Diffusion Model Initialization Time: {:.4f}s'.format(time.time() - start_time))
        self.logger.divider(msg='', end=True)

        return bundle_seq_diffusion_model

    def train(self, epochs: int = 100, learning_rate: float = 0.001, eval_interval: int = 20, early_stop: int = 10):
        start_time = time.time()
        self.logger.divider('Bundle Diffusion Model Training')
        self.logger.info('Training Parameters: {}'.format({'epochs': epochs, 'learning_rate': learning_rate}))

        user_bundle_item_knowledge_graph = self.bundle_seq_data_processor.build_knowledge_graph('user-bundle-item')

        best_precision = -1
        best_model = None
        early_stop_counter = 0

        optimizer = torch.optim.Adam(self.bundle_seq_diffusion_model.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            self.bundle_seq_diffusion_model.train()

            total_loss = self.train_one_epoch(optimizer, user_bundle_item_knowledge_graph)

            if epoch % eval_interval == 0 or early_stop_counter == early_stop:
                total_hit, total = self.evaluate(epoch=epoch, total_loss=total_loss)
                precision = total_hit / total
                early_stop_counter += 1

                if precision > best_precision:
                    best_precision = precision
                    best_model = self.bundle_seq_diffusion_model.state_dict()
                    early_stop_counter = 0

            if early_stop_counter > early_stop:
                break

        self.bundle_seq_diffusion_model.load_state_dict(best_model)

        self.logger.info('Bundle Diffusion Model Training Time: {:.4f}s'.format(time.time() - start_time))
        self.logger.divider(msg='', end=True)

    def train_one_epoch(
            self,
            optimizer: torch.optim.Optimizer,
            user_bundle_item_knowledge_graph: torch_geometric.data.Data
    ) -> torch.Tensor:
        self.bundle_seq_diffusion_model.train()
        total_loss = torch.tensor(0.0, requires_grad=True).to(self.params['device'])
        for sample in self.bundle_seq_train_dataloader:
            user, user_items, prev_bundles, prev_bundle_items, prev_user_bundle_overlap_items, augment_bundles = sample
            diffusion_loss = self.bundle_seq_diffusion_model.forward_diffusion(
                user=user,
                user_items=user_items,
                prev_bundles=prev_bundles,
                prev_bundle_items=prev_bundle_items,
                prev_user_bundle_overlap_items=prev_user_bundle_overlap_items,
                augment_bundles=augment_bundles,
                user_bundle_item_knowledge_graph=user_bundle_item_knowledge_graph
            )
            optimizer.zero_grad()
            total_loss = total_loss + diffusion_loss
            diffusion_loss.backward()
            optimizer.step()

        return total_loss

    def evaluate(self, epoch: int, total_loss: torch.Tensor) -> (int, int):
        self.bundle_seq_diffusion_model.eval()
        total_hit = 0
        total = 0
        show_size = 10
        show_pred_result = True

        self.logger.divider('Epoch: {}, Total Loss: {}'.format(epoch, total_loss.item()))

        for sample in self.bundle_seq_train_dataloader:
            user, user_items, prev_bundles, prev_bundle_items, prev_user_bundle_overlap_items, augment_bundles = sample
            augment_bundles_embedding = self.bundle_seq_diffusion_model.reverse_diffusion(
                user=user,
                user_items=user_items,
                prev_bundles=prev_bundles,
                prev_bundle_items=prev_bundle_items,
                prev_user_bundle_overlap_items=prev_user_bundle_overlap_items
            )

            distances = torch.cdist(augment_bundles_embedding, self.bundle_seq_diffusion_model.bundle_embedding.weight)
            pred_bundles = torch.argmin(distances, dim=-1)

            if show_pred_result:
                show_size = min(show_size, augment_bundles.shape[0])
                self.logger.info('User ID: {}'.format(user[:show_size].tolist()))
                self.logger.info('True Augment Bundle IDs: {}'.format(augment_bundles[:show_size].tolist()))
                self.logger.info('Pred Augment Bundle IDs: {}'.format(pred_bundles[:show_size].tolist()))
                show_pred_result = False

            # current_hit = (augment_bundles.unsqueeze(2) == pred_bundles.unsqueeze(1)).sum(dim=2).sum(dim=1)
            # current_hit_label = current_hit.to(torch.bool)
            #
            # _, pred_bundles_index, pred_bundles_duplicate_count = pred_bundles.unsqueeze(1).unique_consecutive(
            #     return_inverse=True, return_counts=True
            # )
            # pred_bundles_duplicate_count = pred_bundles_duplicate_count[pred_bundles_index][:, 0, 0] - 1
            # pred_bundles_duplicate_count = (current_hit_label * pred_bundles_duplicate_count).sum().item()
            #
            # current_hit_count = current_hit.sum().item() - pred_bundles_duplicate_count

            # total_hit += current_hit_count
            # total += augment_bundles.shape[0] * augment_bundles.shape[1]

            for i in range(len(user)):
                current_hit = torch.isin(augment_bundles[i], pred_bundles[i])
                current_hit_count = current_hit.sum().item()
                total_hit += current_hit_count
                total += augment_bundles.shape[1]

        self.logger.divider(msg='Hit: {}, Total: {}, Precision: {}'.format(total_hit, total, total_hit / total), end=True)

        return total_hit, total

    def save_model(self):
        self.logger.divider('Bundle Seq Diffusion Model Saving')
        model_name = '{}.pth'.format(self.prefix)
        model_save_path = os.path.join(self.output_path, model_name)

        torch.save(self.bundle_seq_diffusion_model.state_dict(), model_save_path)
        self.logger.info('Bundle Seq Diffusion Model Saved: {}'.format(model_save_path))
        self.logger.divider(msg='', end=True)

    def augmentation(
            self,
            augment_bundle_interaction_num_range_list: list,
            augment_dataset_dataloader_list: torch.utils.data.DataLoader
    ):
        start_time = time.time()
        self.logger.divider('Bundle Seq Dataset Augmentation')
        self.logger.info(
            'Augment User-Bundle Interaction Data for users, witch bundle interactions in the range of {}'.format(
                augment_bundle_interaction_num_range_list
            )
        )

        augment_results = {}

        for augment_dataset_dataloader in augment_dataset_dataloader_list:

            self.bundle_seq_diffusion_model.eval()
            for sample in augment_dataset_dataloader:
                user, user_items, prev_bundles, prev_bundle_items, prev_user_bundle_overlap_items, _ = sample
                augment_bundles_embedding = self.bundle_seq_diffusion_model.reverse_diffusion(
                    user=user,
                    user_items=user_items,
                    prev_bundles=prev_bundles,
                    prev_bundle_items=prev_bundle_items,
                    prev_user_bundle_overlap_items=prev_user_bundle_overlap_items
                )

                distances = torch.cdist(augment_bundles_embedding, self.bundle_seq_diffusion_model.bundle_embedding.weight)
                pred_bundles = torch.argmin(distances, dim=-1)

                for i in range(len(user)):
                    augment_results.setdefault(user[i].item(), []).extend(pred_bundles[i].tolist())

        for user, bundles in augment_results.items():
            augment_results[user] = list(set(bundles))

        # augment_results to pd.DataFrame
        augment_results_df = []
        for user, bundles in augment_results.items():
            for bundle in bundles:
                augment_results_df.append([user, bundle])

        augment_results_df = pd.DataFrame(augment_results_df, columns=['user_id', 'bundle_id'])

        # calculate the overlap between augment_results_df and ground truth
        overlap_data = pd.merge(
            augment_results_df, self.bundle_seq_data_processor.ground_true_user_bundle_data,
            how='inner',
            on=['user_id', 'bundle_id']
        )
        self.logger.info('Augmentation Overlap with Ground Truth: {}'.format(overlap_data.shape[0]))

        # add to raw train data
        raw_train_data = self.bundle_seq_data_processor.user_bundle_data
        raw_train_data = raw_train_data.append(augment_results_df, ignore_index=True)

        # save raw_train_data to txt file each line is user_id \t bundle_id
        augment_result_save_path = os.path.join(
            self.bundle_seq_data_processor.dataset_root_path, '{}.txt'.format(self.prefix)
        )
        with open(augment_result_save_path, 'w') as f:
            for i in range(len(raw_train_data)):
                f.write('{}\t{}\n'.format(raw_train_data.iloc[i]['user_id'], raw_train_data.iloc[i]['bundle_id']))

        self.logger.info('Before Augmentation, User-Bundle Interaction Data Shape: {}'.format(self.bundle_seq_data_processor.user_bundle_data.shape))
        self.logger.info('After Augmentation, User-Bundle Interaction Data Shape: {}'.format(raw_train_data.shape))
        self.logger.info('Augmented User-Bundle Interaction Data Saved: {}'.format(augment_result_save_path))
        self.logger.info('Augmentation Done, Cost time: {:.2f}'.format(time.time() - start_time))
        self.logger.divider(msg='', end=True)
