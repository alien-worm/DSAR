import torch


class MLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(self.input_dim, self.hidden_dim))
        for _ in range(self.num_layers - 1):
            self.layers.append(torch.nn.Linear(self.hidden_dim, self.hidden_dim))
        self.layers.append(torch.nn.Linear(self.hidden_dim, self.output_dim))

        self.dropout = torch.nn.Dropout(self.dropout)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_layers):
            x = self.layers[i](x)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.layers[-1](x)
        return x
