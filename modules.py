import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, num_layers, layer_dim, pre_batchnorm=False, pre_activation=False, batch_norm=True):  # num_layers: int; layer_dim: list; num_layers = len(layer_dim) - 1
        super().__init__()
        modules = []
        assert num_layers > 0
        if pre_batchnorm:
            modules.append(nn.BatchNorm1d(layer_dim[0]))
        if pre_activation:
            modules.append(nn.ReLU())
        for i in range(1, num_layers + 1):
            modules.append(nn.Linear(layer_dim[i - 1], layer_dim[i]))
            if i < num_layers:
                if batch_norm:
                    modules.append(nn.BatchNorm1d(layer_dim[i]))
                modules.append(nn.ReLU())
        self._mlp = nn.Sequential(*modules)
        self.dim_in = layer_dim[0]
        self.dim_out = layer_dim[-1]

    def forward(self, x):  # x.size=(batch_size, dim_state)
        output = self._mlp(x)
        return output


class CGBlock(nn.Module):
    def __init__(self, num_mlp_layers, mlp_layer_dim):
        super().__init__()
        self.s_mlp = MLP(num_mlp_layers, mlp_layer_dim)
        self.c_mlp = MLP(num_mlp_layers, mlp_layer_dim)
        self.dim_in = self.s_mlp.dim_in
        self.dim_out = self.s_mlp.dim_out

    def forward(self, pooling, s, c):  # s.size=(batch_size, num_time_step, dim_state), c.size=(batch_size, dim_state)
        s_temp = []
        assert len(s.shape) == 3 and len(c.shape) == 2
        c = self.c_mlp(c)
        for i in range(s.size(dim=1)):
            s_i = self.s_mlp(s[:, i, :])
            s_i = s_i * c
            s_temp.append(s_i)
        s_out = torch.transpose(torch.stack(s_temp), 0, 1)
        if pooling == "max":
            aggregated_c = torch.max(s_out, dim=1, keepdim=False)[0]
        elif pooling == "mean":
            aggregated_c = torch.mean(s_out, dim=1, keepdim=False)
        else:
            raise Exception("Unknown pooling mode for CG")
        return s_out, aggregated_c


class MCGBlock(nn.Module):
    def __init__(self, num_cgblocks, num_mlp_layers_list, mlp_layer_dim_list):
        super().__init__()
        self._blocks = []
        for i in range(num_cgblocks):
            self._blocks.append(CGBlock(num_mlp_layers_list[i], mlp_layer_dim_list[i]))
        self._blocks = nn.ModuleList(self._blocks)
        self.dim_in = self._blocks[0].dim_in
        self.dim_out = self._blocks[-1].dim_out

    def _compute_running_mean(self, prevoius_mean, new_value, i):
        result = (prevoius_mean * i + new_value) / (i + 1)
        return result

    def forward(self, s, c=None, return_s=False):
        assert s.size(dim=2) == self.dim_in
        if c is None:
            c = torch.ones(s.size(dim=0), s.size(dim=2), requires_grad=True)
        assert torch.isfinite(s).all()
        assert torch.isfinite(c).all()
        running_mean_s, running_mean_c = s, c
        for i, cg_block in enumerate(self._blocks, start=1):
            s, c = cg_block("max", running_mean_s, running_mean_c)
            assert torch.isfinite(s).all()
            assert torch.isfinite(c).all()
            running_mean_s = self._compute_running_mean(running_mean_s, s, i)
            running_mean_c = self._compute_running_mean(running_mean_c, c, i)
            assert torch.isfinite(running_mean_s).all()
            assert torch.isfinite(running_mean_c).all()
        if return_s:
            return running_mean_s, running_mean_c
        return running_mean_c


class Encoder(nn.Module):
    def __init__(self, dim_in, dim_hidden):
        super().__init__()
        self.lstm = nn.LSTM(dim_in, dim_hidden, batch_first=True)
        self.mcg = MCGBlock(2, [3, 3], [[dim_hidden, dim_hidden * 2, dim_hidden * 2, dim_hidden], [dim_hidden, dim_hidden * 2, dim_hidden * 2, dim_hidden]])
        self.dim_in = dim_in
        self.dim_out = dim_hidden

    def forward(self, x):
        lstm_output, _ = self.lstm(x)
        c_output = self.mcg(lstm_output)
        return c_output


class Decoder(nn.Module):
    def __init__(self, dim_context):
        super().__init__()
        self.lstm = nn.LSTM(1, dim_context, batch_first=True)
        self.mcg = MCGBlock(2, [3, 3], [[dim_context, dim_context * 2, dim_context * 2, dim_context], [dim_context, dim_context * 2, dim_context * 2, dim_context]])
        self.mlp = MLP(3, [dim_context, dim_context * 2, dim_context, 1])

    def forward(self, s, c):  # s.size = (batch_size, num_time_step, 1), c.size = (batch_size, dim_context)
        assert torch.isfinite(s).all()
        assert torch.isfinite(c).all()
        h, _ = self.lstm(s)
        h_out, _ = self.mcg(h, c, return_s=True)
        output = self.mlp(torch.reshape(h_out, (-1, h_out.shape[-1])))
        output = torch.reshape(output, (h_out.shape[0], h_out.shape[1], -1))
        return output
