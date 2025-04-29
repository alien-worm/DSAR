import os
import sqlite3
import time
import pandas as pd
import numpy as np

import torch
import torch_geometric

from tqdm import tqdm

from utils.logger import Logger
from utils.data_util import text_to_dataframe


class BundleSeqDataset(torch.utils.data.Dataset):

    def __init__(self, data: list):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class BundleSeqDataProcessor(object):

    def __init__(
            self,
            dataset_name: str,
            dataset_root_path: str,
            train_bundle_interaction_num_range: tuple,
            augment_bundle_interaction_num_range_list: list,
            max_sample_num_per_user: int = 1,
            max_user_bundle_interaction_num_quantile: float = 1.0,
            max_user_item_interaction_num_quantile: float = 1.0,
            max_bundle_item_interaction_num_quantile: float = 1.0,
            augment_num: int = None,
            device: str = 'cuda',
            logger: Logger = None
    ):
        self.dataset_name = dataset_name
        self.dataset_root_path = dataset_root_path
        self.train_bundle_interaction_num_range = train_bundle_interaction_num_range
        self.augment_bundle_interaction_num_range_list = augment_bundle_interaction_num_range_list

        max_ubi_num = train_bundle_interaction_num_range[0]
        self.max_sample_num_per_user = min(
            np.math.factorial(max_ubi_num) // np.math.factorial(max_ubi_num - augment_num),
            max_sample_num_per_user
        )

        self.max_user_bundle_interaction_num_quantile = max_user_bundle_interaction_num_quantile
        self.max_user_item_interaction_num_quantile = max_user_item_interaction_num_quantile
        self.max_bundle_item_interaction_num_quantile = max_bundle_item_interaction_num_quantile
        self.augment_num = augment_num
        self.device = device
        self.logger = logger if logger is not None else Logger()

        self.dataset_root_path = os.path.join(self.dataset_root_path, self.dataset_name)

        self.user_bundle_data, self.ground_true_user_bundle_data, self.user_item_data, self.bundle_item_data = self._load_raw_data()

        self.train_user_bundle_data = self._filter_user_bundle_data(
            bundle_interaction_num_range=train_bundle_interaction_num_range
        )
        self.augment_user_bundle_data_dict = {
            augment_bundle_interaction_num_range: self._filter_user_bundle_data(
                bundle_interaction_num_range=augment_bundle_interaction_num_range
            )
            for augment_bundle_interaction_num_range in augment_bundle_interaction_num_range_list
        }
        self.filtered_user_item_data, self.filtered_bundle_item_data = self._filter_user_item_data()

        node_num_info, max_node_info, max_interaction_info = self._get_data_size_info()
        self.user_num, self.bundle_num, self.item_num = node_num_info
        self.max_user_id, self.max_bundle_id, self.max_item_id = max_node_info
        self.max_user_bundle_interaction_num, self.max_user_item_interaction_num, self.max_bundle_item_interaction_num = max_interaction_info

        self.user_bundle_overlap_item = self._get_user_bundle_overlap_item()

    def _load_raw_data(self) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        start_time = time.time()
        self.logger.divider('Start loading raw data')

        user_bundle_data = text_to_dataframe(
            file_path=os.path.join(self.dataset_root_path, 'user_bundle_train.txt'),
            columns=['user_id', 'bundle_id']
        )
        self.logger.info('User bundle train data shape: {}'.format(user_bundle_data.shape))

        # the test data is used to calculate the overlap with the augmented data
        ground_true_user_bundle_data = text_to_dataframe(
            file_path=os.path.join(self.dataset_root_path, 'user_bundle_test.txt'),
            columns=['user_id', 'bundle_id']
        )
        self.logger.info('User bundle test data shape: {}'.format(ground_true_user_bundle_data.shape))

        user_item_data = text_to_dataframe(
            file_path=os.path.join(self.dataset_root_path, 'user_item.txt'),
            columns=['user_id', 'item_id']
        )
        self.logger.info('User item data shape: {}'.format(user_item_data.shape))

        bundle_item_data = text_to_dataframe(
            file_path=os.path.join(self.dataset_root_path, 'bundle_item.txt'),
            columns=['bundle_id', 'item_id']
        )
        self.logger.info('Bundle item data shape: {}'.format(bundle_item_data.shape))

        self.logger.info('End loading raw data, Cost time: {:.2f}'.format(time.time() - start_time))
        self.logger.divider(msg='', end=True)

        return user_bundle_data, ground_true_user_bundle_data, user_item_data, bundle_item_data

    def _filter_user_bundle_data(self, bundle_interaction_num_range: tuple) -> pd.DataFrame:
        start_time = time.time()
        self.logger.divider('Start filtering user bundle data')

        bundle_interactions = self.user_bundle_data.groupby(['user_id']).size().reset_index(name='count')

        bundle_interactions = bundle_interactions[
            (bundle_interactions['count'] >= bundle_interaction_num_range[0]) &
            (bundle_interactions['count'] <= bundle_interaction_num_range[1])
            ]
        user_ids = bundle_interactions['user_id'].values
        user_bundle_data = self.user_bundle_data[self.user_bundle_data['user_id'].isin(user_ids)]
        self.logger.info('User bundle data shape: {}'.format(user_bundle_data.shape))

        self.logger.info('End filtering user bundle data, Cost time: {:.2f}'.format(time.time() - start_time))
        self.logger.divider(msg='', end=True)

        return user_bundle_data

    def _filter_user_item_data(self) -> (pd.DataFrame, pd.DataFrame):
        start_time = time.time()
        self.logger.divider('Start filtering user item data')

        user_list = self.train_user_bundle_data['user_id'].unique().tolist()
        for _, augment_user_bundle_data in self.augment_user_bundle_data_dict.items():
            user_list.extend(augment_user_bundle_data['user_id'].unique().tolist())
        user_list = list(set(user_list))

        filtered_user_item_data = self.user_item_data[self.user_item_data['user_id'].isin(user_list)]
        self.logger.info('Filtered user item data shape: {}'.format(filtered_user_item_data.shape))

        bundle_list = self.train_user_bundle_data['bundle_id'].unique().tolist()
        for _, augment_user_bundle_data in self.augment_user_bundle_data_dict.items():
            user_list.extend(augment_user_bundle_data['bundle_id'].unique().tolist())
        bundle_list = list(set(bundle_list))

        filtered_bundle_item_data = self.bundle_item_data[self.bundle_item_data['bundle_id'].isin(bundle_list)]
        self.logger.info('Filtered bundle item data shape: {}'.format(filtered_bundle_item_data.shape))

        self.logger.info('End filtering user item data, Cost time: {:.2f}'.format(time.time() - start_time))
        self.logger.divider(msg='', end=True)

        return filtered_user_item_data, filtered_bundle_item_data

    def _get_data_size_info(self) -> ((int, int, int), (int, int, int), (int, int, int)):
        """
        Get data size info
        :return:
            (user_num, bundle_num, item_num),
            (max_user_id, max_bundle_id, max_item_id),
            (max_user_bundle_interaction_num, max_user_item_interaction_num, max_bundle_item_interaction_num)
        """

        self.logger.divider('Start getting data size')
        data_size = text_to_dataframe(
            file_path=os.path.join(self.dataset_root_path, '{}_data_size.txt'.format(self.dataset_name)),
            columns=['user_num', 'bundle_num', 'item_num']
        )
        user_num, bundle_num, item_num = data_size.iloc[0].values
        self.logger.info('User num: {}, Bundle num: {}, Item num: {}'.format(user_num, bundle_num, item_num))

        max_user_id, max_bundle_id, max_item_id = user_num, user_num + bundle_num, user_num + bundle_num + item_num
        self.logger.info('Max user id: {}, Max bundle id: {}, Max item id: {}'.format(
            max_user_id, max_bundle_id, max_item_id
        ))

        max_augment_user_bundle_interaction_num = -1
        for _, augment_user_bundle_data in self.augment_user_bundle_data_dict.items():
            max_augment_user_bundle_interaction_num = max(
                max_augment_user_bundle_interaction_num,
                augment_user_bundle_data.groupby('user_id').size().quantile(
                    self.max_user_bundle_interaction_num_quantile
                )
            )

        max_user_bundle_interaction_num = int(
            max(
                self.train_user_bundle_data.groupby('user_id').size().quantile(
                    self.max_user_bundle_interaction_num_quantile
                ),
                max_augment_user_bundle_interaction_num
            )
        )
        max_user_item_interaction_num = int(
            self.filtered_user_item_data.groupby('user_id').size().quantile(
                self.max_user_item_interaction_num_quantile
            )
        )
        max_bundle_item_interaction_num = int(
            self.filtered_bundle_item_data.groupby('bundle_id').size().quantile(
                self.max_bundle_item_interaction_num_quantile
            )
        )

        self.logger.info('Max user bundle interaction num: {}'.format(max_user_bundle_interaction_num))
        self.logger.info('Max user item interaction num: {}'.format(max_user_item_interaction_num))
        self.logger.info('Max bundle item interaction num: {}'.format(max_bundle_item_interaction_num))

        self.logger.divider(msg='', end=True)

        return (user_num, bundle_num, item_num), \
            (max_user_id, max_bundle_id, max_item_id), \
            (max_user_bundle_interaction_num, max_user_item_interaction_num, max_bundle_item_interaction_num)

    def _get_user_bundle_overlap_item(self) -> dict:
        start_time = time.time()
        self.logger.divider('Start getting user bundle overlap item')

        sqlite_conn = sqlite3.connect(':memory:')

        self.filtered_user_item_data.to_sql('user_item', sqlite_conn, index=False)
        self.filtered_bundle_item_data.to_sql('bundle_item', sqlite_conn, index=False)

        query = """
            SELECT user_id, bundle_id, GROUP_CONCAT(user_item.item_id, ',') AS item_ids
            FROM user_item
            JOIN bundle_item ON user_item.item_id = bundle_item.item_id
            GROUP BY user_id, bundle_id;
        """

        cursor = sqlite_conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) 
            FROM (
                SELECT user_id, bundle_id 
                FROM user_item JOIN bundle_item 
                ON user_item.item_id = bundle_item.item_id 
                GROUP BY user_id, bundle_id
            ) as grouped_table;
        """)
        total_rows = cursor.fetchone()[0]

        user_bundle_overlap_item = pd.DataFrame()
        with tqdm(total=total_rows, desc='Getting user bundle overlap item') as progress_bar:
            for chunk in pd.read_sql_query(query, sqlite_conn, chunksize=100000):
                user_bundle_overlap_item = pd.concat([user_bundle_overlap_item, chunk])
                progress_bar.update(len(chunk))

        user_bundle_overlap_item = user_bundle_overlap_item.set_index(['user_id', 'bundle_id'])['item_ids'].apply(
            lambda x: np.array(x.split(',')).astype(int) + self.user_num + self.bundle_num
        ).to_dict()

        user_item_padding = [self.max_item_id] * self.max_user_item_interaction_num
        user_bundle_overlap_item = {
            (user_id, bundle_id): list(item_list) + user_item_padding[len(item_list):]
            for (user_id, bundle_id), item_list in tqdm(user_bundle_overlap_item.items(), desc='Padding')
        }

        sqlite_conn.close()

        self.logger.info('End getting user bundle overlap item, Cost time: {:.2f}'.format(time.time() - start_time))
        self.logger.divider(msg='', end=True)

        return user_bundle_overlap_item

    def _build_user_bundle_item_knowledge_graph(self) -> torch_geometric.data.Data:
        # user id range: [0, user_num)
        # bundle id range: [user_num, user_num + bundle_num)
        # item id range: [user_num + bundle_num, user_num + bundle_num + item_num)

        # user bundle edge index
        user_bundle_edge_index = torch.stack([
            torch.tensor(self.user_bundle_data['user_id'].values, dtype=torch.long),
            torch.tensor(self.user_bundle_data['bundle_id'].values + self.user_num, dtype=torch.long)
        ], dim=0)
        user_bundle_edge_type = torch.tensor([0] * len(self.user_bundle_data), dtype=torch.long)

        # bundle item edge index
        bundle_item_edge_index = torch.stack([
            torch.tensor(self.bundle_item_data['bundle_id'].values + self.user_num, dtype=torch.long),
            torch.tensor(self.bundle_item_data['item_id'].values + self.user_num + self.bundle_num, dtype=torch.long)
        ], dim=0)
        bundle_item_edge_type = torch.tensor([1] * len(self.bundle_item_data), dtype=torch.long)

        # user item edge index
        user_item_edge_index = torch.stack([
            torch.tensor(self.user_item_data['user_id'].values, dtype=torch.long),
            torch.tensor(self.user_item_data['item_id'].values + self.user_num + self.bundle_num, dtype=torch.long)
        ], dim=0)
        user_item_edge_type = torch.tensor([2] * len(self.user_item_data), dtype=torch.long)

        user_bundle_item_knowledge_graph = torch_geometric.data.Data(
            edge_index=torch.cat([user_bundle_edge_index, bundle_item_edge_index, user_item_edge_index], dim=1),
            edge_type=torch.cat([user_bundle_edge_type, bundle_item_edge_type, user_item_edge_type], dim=0),
            num_nodes=self.user_num + self.bundle_num + self.item_num,
            num_edges=len(self.user_bundle_data) + len(self.bundle_item_data) + len(self.user_item_data)
        ).to(self.device)

        return user_bundle_item_knowledge_graph

    def build_knowledge_graph(self, graph_type: str = 'user-bundle-item') -> torch_geometric.data.Data:
        start_time = time.time()
        self.logger.divider('Start building knowledge graph')

        if graph_type == 'user-item':
            edge_index = torch.stack([
                torch.tensor(self.user_item_data['user_id'].values, dtype=torch.long, device=self.device),
                torch.tensor(self.user_item_data['item_id'].values + self.user_num, dtype=torch.long,
                             device=self.device)
            ], dim=0)
            num_nodes = self.user_num + self.item_num
            num_edges = len(self.user_item_data)
            edge_type = torch.tensor([0] * num_edges, dtype=torch.long, device=self.device)
        elif graph_type == 'user-bundle':
            edge_index = torch.stack([
                torch.tensor(self.user_bundle_data['user_id'].values, dtype=torch.long, device=self.device),
                torch.tensor(self.user_bundle_data['bundle_id'].values + self.user_num, dtype=torch.long,
                             device=self.device)
            ], dim=0)
            num_nodes = self.user_num + self.bundle_num
            num_edges = len(self.user_bundle_data)
            edge_type = torch.tensor([0] * num_edges, dtype=torch.long, device=self.device)
        elif graph_type == 'bundle-item':
            edge_index = torch.stack([
                torch.tensor(self.bundle_item_data['bundle_id'].values, dtype=torch.long, device=self.device),
                torch.tensor(self.bundle_item_data['item_id'].values + self.bundle_num, dtype=torch.long,
                             device=self.device)
            ], dim=0)
            num_nodes = self.bundle_num + self.item_num
            num_edges = len(self.bundle_item_data)
            edge_type = torch.tensor([0] * num_edges, dtype=torch.long, device=self.device)
        elif graph_type == 'user-bundle-item':
            user_bundle_item_knowledge_graph = self._build_user_bundle_item_knowledge_graph()
            edge_index = user_bundle_item_knowledge_graph.edge_index
            edge_type = user_bundle_item_knowledge_graph.edge_type
            num_nodes = user_bundle_item_knowledge_graph.num_nodes
            num_edges = user_bundle_item_knowledge_graph.num_edges
        else:
            raise ValueError('Invalid graph type, only support user-bundle-item, user-item, user-bundle, bundle-item')

        x = torch.arange(num_nodes, dtype=torch.float, device=self.device).view(-1, 1)
        edge_weight = torch.ones(num_edges, dtype=torch.float, device=self.device)

        knowledge_graph = torch_geometric.data.Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_weight=edge_weight,
            num_nodes=num_nodes,
            num_edges=num_edges
        ).to(self.device)

        self.logger.info(
            'Knowledge graph type: {}, Num nodes: {}, Num edges: {}'.format(graph_type, num_nodes, num_edges))
        self.logger.info('Edge index shape: {}, Edge type shape: {}'.format(edge_index.shape, edge_type.shape))
        self.logger.info('End building knowledge graph, Cost time: {:.2f}'.format(time.time() - start_time))
        self.logger.divider(msg='', end=True)

        return knowledge_graph

    def _data_process(
            self,
            user_bundle_data: pd.DataFrame,
            bundle_interaction_num_range: tuple,
            augment_num: int = 0
    ) -> list:
        start_time = time.time()
        self.logger.divider('Start processing data')
        self.logger.info('Bundle interaction num range: {}'.format(bundle_interaction_num_range))
        self.logger.info('Augment num: {}'.format(augment_num))

        user_bundle_dict = user_bundle_data.groupby('user_id')['bundle_id'].apply(list).to_dict()

        user_item_padding = [self.max_item_id] * self.max_user_item_interaction_num
        bundle_item_padding = [self.max_item_id] * self.max_bundle_item_interaction_num

        user_item_dict = self.filtered_user_item_data.groupby('user_id')['item_id'].apply(list).to_dict()
        user_item_dict = {k: [v + self.user_num + self.bundle_num for v in vs] for k, vs in user_item_dict.items()}
        user_item_dict = {k: vs + user_item_padding[len(vs):] for k, vs in user_item_dict.items()}

        bundle_item_dict = self.filtered_bundle_item_data.groupby('bundle_id')['item_id'].apply(list).to_dict()
        bundle_item_dict = {k: [v + self.user_num + self.bundle_num for v in vs] for k, vs in bundle_item_dict.items()}
        bundle_item_dict = {k: vs + bundle_item_padding[len(vs):] for k, vs in bundle_item_dict.items()}

        prev_bundle_len = bundle_interaction_num_range[1]
        if augment_num != 0:
            prev_bundle_len = bundle_interaction_num_range[1] - augment_num

        processed_data = []
        for user_id, bundle_ids in tqdm(user_bundle_dict.items(), desc='Processing data'):
            user_items = user_item_dict.get(user_id, user_item_padding)[:self.max_user_item_interaction_num]
            bundle_ids = [int(bundle_id) for bundle_id in bundle_ids]

            sample_count = 0
            if augment_num == 0:
                sample_count = self.max_sample_num_per_user - 1

            while sample_count < self.max_sample_num_per_user:

                augment_bundles = []
                prev_bundles = bundle_ids
                if augment_num != 0:
                    # augment_bundles = bundle_ids[-augment_num:]
                    # prev_bundles = bundle_ids[:-augment_num]
                    augment_bundles = np.random.choice(bundle_ids, augment_num, replace=False)
                    prev_bundles = [bundle_id for bundle_id in bundle_ids if bundle_id not in augment_bundles]

                prev_bundles = [self.max_bundle_id] * (
                    prev_bundle_len - len(prev_bundles) if len(prev_bundles) < prev_bundle_len else 0
                ) + prev_bundles

                prev_bundle_items = []
                prev_user_bundle_overlap_items = []
                for prev_bundle_id in prev_bundles:

                    if prev_bundle_id == self.max_bundle_id:
                        prev_bundle_items.append(bundle_item_padding)
                        prev_user_bundle_overlap_items.append(user_item_padding)
                        continue

                    prev_bundle_items.append(
                        bundle_item_dict.get(prev_bundle_id, bundle_item_padding)[:self.max_bundle_item_interaction_num]
                    )
                    prev_user_bundle_overlap_items.append(
                        self.user_bundle_overlap_item.get(
                            (user_id, prev_bundle_id),
                            user_item_padding
                        )[:self.max_user_item_interaction_num]
                    )

                processed_data.append([
                    torch.tensor([int(user_id)], dtype=torch.long, device=self.device),
                    torch.tensor([user_items], dtype=torch.long, device=self.device),
                    torch.tensor(prev_bundles, dtype=torch.long, device=self.device),
                    torch.tensor(prev_bundle_items, dtype=torch.long, device=self.device),
                    torch.tensor(prev_user_bundle_overlap_items, dtype=torch.long, device=self.device),
                    torch.tensor(augment_bundles, dtype=torch.long, device=self.device)
                ])

                sample_count += 1

        self.logger.info('End processing data, Cost time: {:.2f}'.format(time.time() - start_time))
        self.logger.divider(msg='', end=True)

        return processed_data

    def get_train_dataset(self) -> BundleSeqDataset:
        processed_train_data = self._data_process(
            user_bundle_data=self.train_user_bundle_data,
            bundle_interaction_num_range=self.train_bundle_interaction_num_range,
            augment_num=self.augment_num
        )
        return BundleSeqDataset(processed_train_data)

    def get_augment_dataset(self, augment_bundle_interaction_num_range: tuple) -> BundleSeqDataset:
        processed_augment_data = self._data_process(
            user_bundle_data=self.augment_user_bundle_data_dict[augment_bundle_interaction_num_range],
            bundle_interaction_num_range=augment_bundle_interaction_num_range,
            augment_num=0
        )
        return BundleSeqDataset(processed_augment_data)
