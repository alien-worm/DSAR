import torch
import torch_geometric

from transformers import BertConfig, BertModel

from models.mlp import MLP
from models.diffusion_model import DiffusionModel


class ConditionalLayer(torch.nn.Module):

    def __init__(
            self,
            user_num: int,
            bundle_num: int,
            item_num: int,
            embedding_dim: int
    ):
        super(ConditionalLayer, self).__init__()
        self.user_num = user_num
        self.bundle_num = bundle_num
        self.item_num = item_num
        self.embedding_dim = embedding_dim

        # take user, user-item, user-bundle, bundle-item interaction as input
        self.mlp_layer = MLP(
            input_dim=embedding_dim * 5,
            hidden_dim=embedding_dim * 5,
            output_dim=embedding_dim,
            num_layers=5,
            dropout=0
        )

    @staticmethod
    def node_embedding_aggregation(
            node_ids: torch.Tensor,
            max_node_id: int,
            user_bundle_item_graph_conv: torch.nn.Module
    ) -> torch.Tensor:
        """
        :param node_ids: node ids, shape: (batch_size, num_nodes)
        :param max_node_id: max node number
        :param user_bundle_item_graph_conv: user-bundle-item graph convolutional layer
        :return:
        """
        node_embedding_mask = torch.where(
            node_ids == max_node_id,
            torch.tensor(0, device=node_ids.device),
            torch.tensor(1, device=node_ids.device)
        )
        node_embedding = user_bundle_item_graph_conv.embedding(node_ids) * node_embedding_mask.unsqueeze(-1)
        node_embedding = torch.sum(node_embedding, dim=1)

        if len(node_embedding.shape) == 3:
            node_embedding = torch.sum(node_embedding, dim=1)

        return node_embedding

    def forward(
            self,
            user: torch.Tensor,
            user_items: torch.Tensor,
            prev_bundles: torch.Tensor,
            prev_bundle_items: torch.Tensor,
            prev_user_bundle_overlap_items: torch.Tensor,
            user_bundle_item_graph_conv: torch.nn.Module
    ) -> torch.Tensor:
        """
        :param user: user id, shape: (batch_size, 1)
        :param user_items: item ids interacted by user, shape: (batch_size, 1, max_user_item_interaction)
        :param prev_bundles: bundle ids interacted by user, shape: (batch_size, max_user_bundle_interaction)
        :param prev_bundle_items: item ids in bundle, shape: (batch_size, max_user_bundle_interaction, max_user_bundle_interaction)
        :param prev_user_bundle_overlap_items: item ids interacted by user in bundle, shape: (batch_size, max_user_bundle_interaction, max_user_bundle_interaction)
        :param user_bundle_item_graph_conv: user-bundle-item graph convolutional layer
        :return:
        """

        # get user embedding
        user_embedding = user_bundle_item_graph_conv.embedding(user)
        user_embedding = torch.squeeze(user_embedding, dim=1)

        # get user-items embedding
        user_items_embedding = self.node_embedding_aggregation(
            node_ids=user_items,
            max_node_id=self.user_num + self.bundle_num + self.item_num,
            user_bundle_item_graph_conv=user_bundle_item_graph_conv
        )

        # get prev-bundles embedding
        prev_bundles_embedding = self.node_embedding_aggregation(
            node_ids=prev_bundles,
            max_node_id=self.user_num + self.bundle_num,
            user_bundle_item_graph_conv=user_bundle_item_graph_conv
        )

        # get item embedding in bundle
        prev_bundle_items_embedding = self.node_embedding_aggregation(
            node_ids=prev_bundle_items,
            max_node_id=self.user_num + self.item_num + self.bundle_num,
            user_bundle_item_graph_conv=user_bundle_item_graph_conv
        )

        # get item overlap embedding
        prev_user_bundle_overlap_items_embedding = self.node_embedding_aggregation(
            node_ids=prev_user_bundle_overlap_items,
            max_node_id=self.user_num + self.item_num + self.bundle_num,
            user_bundle_item_graph_conv=user_bundle_item_graph_conv
        )

        # concatenate user, item, bundle, item in bundle
        x = torch.cat(
            [user_embedding, user_items_embedding, prev_bundles_embedding, prev_bundle_items_embedding,
             prev_user_bundle_overlap_items_embedding], dim=-1
        )

        return self.mlp_layer(x)


class DenoisingLayer(torch.nn.Module):

    def __init__(self, bert_config: BertConfig, embedding_dim: int):
        super(DenoisingLayer, self).__init__()
        self.bert_model = BertModel(bert_config)
        # self.linear_layer = torch.nn.Linear(embedding_dim, embedding_dim)
        self.mlp_layer = MLP(
            input_dim=embedding_dim,
            hidden_dim=embedding_dim * 2,
            output_dim=embedding_dim,
            num_layers=5,
            dropout=0
        )

    def forward(self, x: torch.Tensor, signal: torch.Tensor) -> torch.Tensor:
        signal = signal.unsqueeze(1)
        x = x + signal
        x = self.bert_model(inputs_embeds=x).last_hidden_state
        x = self.mlp_layer(x)
        return x


class BundleSeqDiffusionModel(DiffusionModel):

    def __init__(
            self,
            user_num: int,
            bundle_num: int,
            item_num: int,
            augment_num: int,
            lightgcn_layers: int = 3,
            noise_schedule: str = 'linear',
            noise_scale: float = 0.1,
            min_noise: float = 0.01,
            max_noise: float = 0.1,
            max_diffusion_steps: int = 10,
            embedding_dim: int = 128,
            bert_config: BertConfig = None,
            device: str = 'cpu'
    ):
        super(BundleSeqDiffusionModel, self).__init__(
            noise_schedule=noise_schedule,
            noise_scale=noise_scale,
            min_noise=min_noise,
            max_noise=max_noise,
            max_diffusion_steps=max_diffusion_steps,
            embedding_dim=embedding_dim,
            device=device
        )

        # User, Item, Bundle Embedding
        self.user_num = user_num
        self.bundle_num = bundle_num
        self.item_num = item_num
        self.augment_num = augment_num
        self.bundle_embedding = torch.nn.Embedding(bundle_num, embedding_dim)

        # user id range: [0, user_num)
        # bundle id range: [user_num, user_num + bundle_num)
        # item id range: [user_num + bundle_num, user_num + bundle_num + item_num)
        self.user_bundle_item_graph_conv = torch_geometric.nn.LightGCN(
            num_nodes=user_num + bundle_num + item_num + 1,
            embedding_dim=embedding_dim,
            num_layers=lightgcn_layers
        )

        # define conditional layer
        self.conditional_layer = ConditionalLayer(
            user_num=user_num,
            bundle_num=bundle_num,
            item_num=item_num,
            embedding_dim=embedding_dim
        )

        # define denoising layer
        if bert_config is None:
            bert_config = BertConfig(
                hidden_size=embedding_dim,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=embedding_dim * 2
            )
        self.denoising_layer = DenoisingLayer(bert_config, embedding_dim)

    def forward_diffusion(
            self,
            user: torch.Tensor,
            user_items: torch.Tensor,
            prev_bundles: torch.Tensor,
            prev_bundle_items: torch.Tensor,
            prev_user_bundle_overlap_items: torch.Tensor,
            augment_bundles: torch.Tensor,
            user_bundle_item_knowledge_graph: torch_geometric.data.Data
    ) -> torch.Tensor:
        """
        :param user: user id, shape: (batch_size, 1)
        :param user_items: item ids interacted by user, shape: (batch_size, 1, max_user_item_interaction)
        :param prev_bundles: bundle ids interacted by user, shape: (batch_size, max_user_bundle_interaction)
        :param prev_bundle_items: item ids in bundle, shape: (batch_size, max_user_bundle_interaction, max_user_bundle_interaction)
        :param prev_user_bundle_overlap_items: item ids interacted by user in bundle, shape: (batch_size, max_user_bundle_interaction, max_user_bundle_interaction)
        :param augment_bundles: bundle ids to augment, shape: (batch_size, augment_num)
        :param user_bundle_item_knowledge_graph: user-bundle-item knowledge graph
        :return: mse loss
        """

        # sample diffusion steps
        diffusion_steps = torch.randint(
            low=0,
            high=self.max_diffusion_steps,
            size=(user.size(0),),
            device=self.device
        )
        diffusion_steps_embedding = self.diffusion_step_embedding(diffusion_steps)

        # generate conditional signal
        self.user_bundle_item_graph_conv(user_bundle_item_knowledge_graph.edge_index)
        conditional_signal = self.conditional_layer(
            user=user,
            user_items=user_items,
            prev_bundles=prev_bundles,
            prev_bundle_items=prev_bundle_items,
            prev_user_bundle_overlap_items=prev_user_bundle_overlap_items,
            user_bundle_item_graph_conv=self.user_bundle_item_graph_conv
        )

        # add diffusion steps embedding to conditional signal
        conditional_signal = conditional_signal + diffusion_steps_embedding

        # add noise to target bundle embedding
        augment_bundles_embedding = self.bundle_embedding(augment_bundles)

        noised_augment_bundles_embedding = self.add_gaussian_noise(
            x=augment_bundles_embedding,
            diffusion_steps=diffusion_steps
        )

        # use conditional signal to denoise noised target bundle embedding
        denoised_augment_bundles_embedding = self.denoising_layer(
            x=noised_augment_bundles_embedding,
            signal=conditional_signal
        )

        # calculate loss
        loss = torch.nn.functional.mse_loss(
            denoised_augment_bundles_embedding,
            augment_bundles_embedding
        )

        return loss

    def reverse_diffusion(
            self,
            user: torch.Tensor,
            user_items: torch.Tensor,
            prev_bundles: torch.Tensor,
            prev_bundle_items: torch.Tensor,
            prev_user_bundle_overlap_items: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param user: user id, shape: (batch_size, 1)
        :param user_items: item ids interacted by user, shape: (batch_size, 1, max_user_item_interaction)
        :param prev_bundles: bundle ids interacted by user, shape: (batch_size, max_user_bundle_interaction)
        :param prev_bundle_items: item ids in bundle, shape: (batch_size, max_user_bundle_interaction, max_user_bundle_interaction)
        :param prev_user_bundle_overlap_items: item ids interacted by user in bundle, shape: (batch_size, max_user_bundle_interaction, max_user_bundle_interaction)
        :return: augment bundles embedding, shape: (batch_size, augment_num, embedding_dim)
        """

        # generate target bundle embedding with gaussian noise
        augment_bundles_embedding = torch.randn(user.shape[0], self.augment_num, self.embedding_dim).to(self.device)

        # generate conditional signal
        conditional_signal = self.conditional_layer(
            user=user,
            user_items=user_items,
            prev_bundles=prev_bundles,
            prev_bundle_items=prev_bundle_items,
            prev_user_bundle_overlap_items=prev_user_bundle_overlap_items,
            user_bundle_item_graph_conv=self.user_bundle_item_graph_conv
        )

        # denoising
        for diffusion_step in range(self.max_diffusion_steps - 1, -1, -1):
            diffusion_step = torch.tensor([diffusion_step] * user.size(0), device=self.device)
            diffusion_step_embedding = self.diffusion_step_embedding(diffusion_step)
            tmp_conditional_signal = conditional_signal + diffusion_step_embedding

            denoised_augment_bundles_embedding = self.denoising_layer(
                x=augment_bundles_embedding,
                signal=tmp_conditional_signal
            )

            augment_bundles_embedding, _, _ = self.p_mean_variance(
                x=augment_bundles_embedding,
                diffusion_steps=diffusion_step,
                model_output=denoised_augment_bundles_embedding
            )

        return augment_bundles_embedding
