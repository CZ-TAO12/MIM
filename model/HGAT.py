from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, Linear
from torch_scatter import scatter_add
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax

class Hyperedge(MessagePassing):
    def __init__(self, **kwargs):
        kwargs.setdefault('aggr', 'add')
        self.heads = 1
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

    def forward(self, x: Tensor, hyperedge_index: Tensor,
                hyperedge_weight: Optional[Tensor] = None) -> Tensor:

        self.out_channels = x.size(-1)

        num_nodes, num_edges = x.size(0), 0
        alpha = None

        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        B = scatter_add(x.new_ones(hyperedge_index.size(1)),
                        hyperedge_index[1], dim=0, dim_size=num_edges)
        B = 1.0 / B
        B[B == float("inf")] = 0

        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,
                             size=(num_nodes, num_edges))

        out = out.view(-1, self.heads * self.out_channels)

        return out

    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:

        H, F = self.heads, self.out_channels

        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out

class HypergraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads= 1,
                 concat=True, negative_slope=0.2, dropout=0, bias=False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels // heads

        self.hyperedge_func = Hyperedge()

        # attention
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin = Linear(in_channels, heads * self.out_channels, bias=False)
        self.lin2 = Linear(in_channels, heads * self.out_channels, bias=False)
        self.lin3 = Linear(in_channels, heads * self.out_channels, bias=False)
        
        self.att = Parameter(torch.Tensor(1, heads, 2 * self.out_channels))
        self.att2 = Parameter(torch.Tensor(1, heads, 2 * self.out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * self.out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        
        glorot(self.att)
        glorot(self.att2)
        zeros(self.bias)
    
    def FFN(self, X):
        output = self.FFN_2(F.relu(self.FFN_1(X)))
        output = F.dropout(output, p=self.dropout, training=self.training)
        return output

    def forward(self, x: Tensor, hyperedge_index: Tensor,
                hyperedge_weight: Optional[Tensor] = None,
                hyperedge_attr: Optional[Tensor] = None) -> Tensor:

        num_nodes, num_edges = x.size(0), 0

        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)
        hyperedge_attr = self.hyperedge_func(x, hyperedge_index)
        

        x = self.lin(x)
        x = x.view(-1, self.heads, self.out_channels)  
        hyperedge_attr = self.lin2(hyperedge_attr)
        hyperedge_attr = hyperedge_attr.view(-1, self.heads, self.out_channels)
        x_i = x[hyperedge_index[0]]
        x_j = hyperedge_attr[hyperedge_index[1]]
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        D = scatter_add(hyperedge_weight[hyperedge_index[1]],
                        hyperedge_index[0], dim=0, dim_size=num_nodes)
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter_add(x.new_ones(hyperedge_index.size(1)),
                        hyperedge_index[1], dim=0, dim_size=num_edges)
        B = 1.0 / B
        B[B == float("inf")] = 0

        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,
                             size=(num_nodes, num_edges)) 

        hyperedge_attr = out.view(-1, self.heads * self.out_channels)
        hyperedge_attr = self.lin3(hyperedge_attr)
        hyperedge_attr = hyperedge_attr.view(-1, self.heads, self.out_channels)
        x_i = x[hyperedge_index[0]]
        x_j = hyperedge_attr[hyperedge_index[1]]
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att2).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = self.propagate(hyperedge_index.flip([0]), x=out, norm=D, alpha=alpha, size=(num_edges, num_nodes))
        
        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)

        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        H, F = self.heads, self.out_channels

        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out


