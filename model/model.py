import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Data
from model.HGAT import HypergraphConv
import torch.nn.init as init
import Constants
from model.TransformerBlock import TransformerBlock
from torch.autograd import Variable

from Optim import ScheduledOptim
from utils.util import *


def get_previous_user_mask(seq, user_size):
    assert seq.dim() == 2
    prev_shape = (seq.size(0), seq.size(1), seq.size(1))
    seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
    previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
    previous_mask = torch.from_numpy(previous_mask)
    if seq.is_cuda:
        previous_mask = previous_mask.cuda()
    masked_seq = previous_mask * seqs.data.float()
    PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
    if seq.is_cuda:
        PAD_tmp = PAD_tmp.cuda()
    masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
    ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
    if seq.is_cuda:
        ans_tmp = ans_tmp.cuda()
    masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float(-1000))
    masked_seq = Variable(masked_seq, requires_grad=False)

    return masked_seq.cuda()

class GraphNN(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.5, is_norm=True):
        super(GraphNN, self).__init__()
        self.embedding = nn.Embedding(ntoken, ninp, padding_idx=0)
        self.gnn1 = GCNConv(ninp, ninp * 2)
        self.gnn2 = GCNConv(ninp * 2, ninp)
        self.is_norm = is_norm

        self.dropout = nn.Dropout(dropout)
        if self.is_norm:
            self.batch_norm = torch.nn.BatchNorm1d(ninp)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.embedding.weight)

    def forward(self, graph):
        graph_edge_index = graph.edge_index.cuda()
        graph_x_embeddings = self.gnn1(self.embedding.weight, graph_edge_index)
        graph_x_embeddings = self.dropout(graph_x_embeddings)
        graph_output = self.gnn2(graph_x_embeddings, graph_edge_index)
        if self.is_norm:
            graph_output = self.batch_norm(graph_output)
        return graph_output.cuda()

class MIMIDP(nn.Module):
    def __init__(self, opt, graph, hypergraphs,  dropout=0.3, reverse=False, attn=False):
        super(MIMIDP, self).__init__()
        self.n_node = opt.n_node
        self.initial_feature = opt.d_model
        self.hidden_size = opt.d_model
        self.pos_dim = opt.pos_dim
        self.drop_r  = opt.dropout
        self.layers = opt.graph_layer
        self.att_head = opt.att_head

        self.reverse= reverse
        self.attn=attn
        self.n_channel = len(hypergraphs) + 1  

        self.dropout = nn.Dropout(dropout)
        self.graph = graph

        self.H_Item = hypergraphs[0]   
        self.H_User =hypergraphs[1]

        self.user_embedding = nn.Embedding(self.n_node, self.initial_feature, padding_idx=0)
        self.weights = nn.ParameterList([nn.Parameter(torch.zeros(self.hidden_size, self.hidden_size)) for _ in range(self.n_channel)])
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(1, self.hidden_size)) for _ in range(self.n_channel)])
        self.att = nn.Parameter(torch.zeros(1, self.hidden_size))
        self.att_m = nn.Parameter(torch.zeros(self.hidden_size, self.hidden_size))
        self.GAT_layers = nn.ModuleList()
        self.HGAT_layers = nn.ModuleList()

        for i in range(self.layers):
            self.GAT_layers.append(GATConv(in_channels = self.hidden_size, out_channels = int(self.hidden_size/self.att_head), heads = self.att_head))
            self.HGAT_layers.append(HypergraphConv(in_channels = self.hidden_size, out_channels = self.hidden_size, heads = self.att_head))
        self.history_ATT = TransformerBlock(input_size=self.hidden_size, n_heads=self.att_head, attn_dropout = self.drop_r)
        self.future_ATT = TransformerBlock(input_size=self.hidden_size, n_heads=self.att_head, attn_dropout = self.drop_r, reverse=True)
        self.reset_parameters()
        self.optimizerAdam = torch.optim.Adam(self.parameters(), betas=(0.9, 0.98), eps= 1e-09)
        self.optimizer = ScheduledOptim(self.optimizerAdam, self.hidden_size, opt.n_warmup_steps)
        self.loss_function = nn.CrossEntropyLoss(size_average=False, ignore_index=0)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def self_gating(self, em, channel):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.weights[channel]) + self.bias[channel]))
    
    def channel_attention(self, *channel_embeddings):
        weights = []
        for embedding in channel_embeddings:
            weights.append(
                torch.sum(
                    torch.multiply(self.att, torch.matmul(embedding, self.att_m)),
                    1))
        embs = torch.stack(weights, dim=0)
        score = F.softmax(embs.t(), dim = -1)
        mixed_embeddings = 0
        for i in range(len(weights)):
            mixed_embeddings += torch.multiply(score.t()[i], channel_embeddings[i].t()).t()
        return mixed_embeddings, score
    
    def _dropout_graph(self, graph, keep_prob):
        edge_attr = graph.edge_attr
        edge_index = graph.edge_index.t()
        
        random_index = torch.rand(edge_attr.shape[0]) + keep_prob
        random_index = random_index.int().bool()

        edge_index = edge_index[random_index]
        edge_attr = edge_attr[random_index]
        return Data(edge_index=edge_index.t(), edge_attr=edge_attr)
    
    def history_cas_learning(self):

        if self.training:
            H_Item = self._dropout_graph(self.H_Item, keep_prob=1-self.drop_r)
            H_User = self._dropout_graph(self.H_User, keep_prob=1-self.drop_r)
            Graph  = self._dropout_graph(self.graph,  keep_prob=1-self.drop_r)
        else:
            H_Item = self.H_Item  
            H_User = self.H_User
            Graph  = self.graph

        H_Item = trans_to_cuda(H_Item)
        H_User = trans_to_cuda(H_User)
        Graph  = trans_to_cuda(Graph)

        u_emb_c1 = self.self_gating(self.user_embedding.weight, 0)
        u_emb_c2 = self.self_gating(self.user_embedding.weight, 1)
        u_emb_c3 = self.self_gating(self.user_embedding.weight, 2)
        all_emb_c1 = [u_emb_c1]
        all_emb_c2 = [u_emb_c2]
        all_emb_c3 = [u_emb_c3]

        for k in range(self.layers):
            u_emb_c1 = self.HGAT_layers[k](u_emb_c1, Graph.edge_index)
            normalize_c1 = F.normalize(u_emb_c1, p=2, dim=1)
            all_emb_c1 += [normalize_c1]
            u_emb_c2 = self.HGAT_layers[k](u_emb_c2, H_Item.edge_index)
            normalize_c2 = F.normalize(u_emb_c2, p=2, dim=1)
            all_emb_c2 += [normalize_c2]

            u_emb_c3 = self.HGAT_layers[k](u_emb_c3, H_User.edge_index)
            normalize_c3 = F.normalize(u_emb_c3, p=2, dim=1)
            all_emb_c3 += [normalize_c3]

        u_emb_c1 = torch.stack(all_emb_c1, dim=1)
        u_emb_c1 = torch.mean(u_emb_c1, dim=1)   
        u_emb_c2 = torch.stack(all_emb_c2, dim=1)
        u_emb_c2 = torch.mean(u_emb_c2, dim=1)
        u_emb_c3 = torch.stack(all_emb_c3, dim=1)
        u_emb_c3 = torch.mean(u_emb_c3, dim=1)
        high_embs, _ = self.channel_attention( u_emb_c1, u_emb_c2, u_emb_c3)
        return high_embs


    def forward(self, input_original, label, input_time, label_time):

        input = input_original

        mask = (input == Constants.PAD)
        mask_label = (label == Constants.PAD)   
        HG_user = self.history_cas_learning()
        diff_emb = F.embedding(input, HG_user)

        past_att_out, past_dist = self.history_ATT(diff_emb, diff_emb, diff_emb, mask=mask.cuda())
        future_emb = F.embedding(label, HG_user)
  
        future_att_out, futrue_dist = self.future_ATT(future_emb, future_emb, future_emb, mask=mask_label.cuda())
        past_output = torch.matmul(past_att_out, torch.transpose(HG_user, 1, 0))
        future_output = torch.matmul(future_att_out, torch.transpose(HG_user, 1, 0))

        mask = get_previous_user_mask(input.cuda(), self.n_node)
        output_past = (past_output + mask).view(-1, past_output.size(-1))

        future_output = future_output.view(-1, past_output.size(-1))
        return output_past, future_output, past_dist, futrue_dist, F.normalize(past_att_out, p=2, dim=1), F.normalize(future_att_out, p=2, dim=1)

    def model_prediction(self, input_original, _):

        input = input_original

        mask = (input == Constants.PAD)
        HG_user = self.history_cas_learning()

        diff_emb = F.embedding(input, HG_user)

        past_att_out, past_dist = self.history_ATT(diff_emb, diff_emb, diff_emb, mask=mask.cuda())

        past_output = torch.matmul(past_att_out, torch.transpose(HG_user, 1, 0))

        mask = get_previous_user_mask(input.cuda(), self.n_node)
        output_past = (past_output + mask).view(-1, past_output.size(-1))

        return output_past
    
    def compute_kl(self, p, q):

        p_loss = F.kl_div(
            F.log_softmax(p+1e-8, dim=-1), F.softmax(q+1e-8, dim=-1), reduction="sum"
        )
        q_loss = F.kl_div(
            F.log_softmax(q+1e-8, dim=-1), F.softmax(p+1e-8, dim=-1), reduction="sum"
        )

        loss = (p_loss + q_loss) / 2
        return loss

    def kl_loss(self, attn, attn_reversed):

        loss = (self.compute_kl(attn.sum(dim=1).view(-1, self.att_head), attn_reversed.sum(dim=1).view(-1, self.att_head))+
                                    self.compute_kl(attn.sum(dim=2).view(-1, self.att_head), attn_reversed.sum(dim=2).view(-1, self.att_head)))/2
        return loss 

    
    def seq2seqloss(self, inp_subseq_encodings: torch.Tensor,
                      label_subseq_encodings: torch.Tensor, input_cas:torch.Tensor) -> torch.Tensor:
        
        sqrt_hidden_size = np.sqrt(self.hidden_size)
        product = torch.mul(inp_subseq_encodings, label_subseq_encodings)
        normalized_dot_product = torch.sum(product, dim=-1) / sqrt_hidden_size 
        numerator = torch.exp(normalized_dot_product) 
        inp_subseq_encodings_trans_expanded = inp_subseq_encodings.unsqueeze(1) 
        label_subseq_encodings_trans = label_subseq_encodings.transpose(1, 2)  
        dot_products = torch.matmul(inp_subseq_encodings_trans_expanded, label_subseq_encodings_trans) 
        dot_products = torch.exp(dot_products / sqrt_hidden_size)
        dot_products = dot_products.sum(-1) 
        denominator = dot_products.sum(1)
        seq2seq_loss_k = -torch.log2(numerator / denominator)
        seq2seq_loss_k = torch.flatten(seq2seq_loss_k)
        input_cas = torch.flatten(input_cas)
        mask = (input_cas != Constants.PAD)
        conf_seq2seq_loss_k = torch.mul(seq2seq_loss_k, mask)
        seq2seq_loss = torch.sum(conf_seq2seq_loss_k)
        return seq2seq_loss
    
   

