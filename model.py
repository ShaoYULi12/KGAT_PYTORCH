import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.softmax import edge_softmax

def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class Aggregator(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, aggregator_type):
        super(Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(dropout)

        if aggregator_type == 'gcn':
            self.W = nn.Linear(self.in_dim, self.out_dim)       # W in Equation (6)
        elif aggregator_type == 'graphsage':
            self.W = nn.Linear(self.in_dim * 2, self.out_dim)   # W in Equation (7)
        elif aggregator_type == 'bi-interaction':
            self.W1 = nn.Linear(self.in_dim, self.out_dim)      # W1 in Equation (8)
            self.W2 = nn.Linear(self.in_dim, self.out_dim)      # W2 in Equation (8)
        else:
            raise NotImplementedError

        self.activation = nn.LeakyReLU()

    def forward(self):
        pass


class KGAT(nn.Module):

    def __init__(self, args,
                 n_users, n_entities, n_relations,
                 user_pre_embed=None, item_pre_embed=None):

        super(KGAT, self).__init__()
        self.use_pretrain = args.use_pretrain

        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations

        self.entity_dim = args.entity_dim   #64
        self.relation_dim = args.relation_dim  #64

        self.aggregation_type = args.aggregation_type

        # args.conv_dim_list: default='[64, 32, 16]',help='Output sizes of every aggregation layer.')
        self.conv_dim_list = [args.entity_dim] + eval(args.conv_dim_list)  

        # [0.1, 0.1, 0.1],help='Dropout probability w.r.t. message dropout for each deep layer. 0: no dropout.')
        self.mess_dropout = eval(args.mess_dropout)

        self.n_layers = len(eval(args.conv_dim_list))

        self.kg_l2loss_lambda = args.kg_l2loss_lambda
        self.cf_l2loss_lambda = args.cf_l2loss_lambda

        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        self.entity_user_embed = nn.Embedding(self.n_entities + self.n_users, self.entity_dim)

        # 对非item的实体进行初始化，然后将user item的预训练embed加入组成entity_user_embed.weight
        if (self.use_pretrain == 1) and (user_pre_embed is not None) and (item_pre_embed is not None):
            other_entity_embed = nn.Parameter(torch.Tensor(self.n_entities - item_pre_embed.shape[0], self.entity_dim))
            nn.init.xavier_uniform_(other_entity_embed, gain=nn.init.calculate_gain('relu'))
            entity_user_embed = torch.cat([item_pre_embed, other_entity_embed, user_pre_embed], dim=0)
            self.entity_user_embed.weight = nn.Parameter(entity_user_embed)

        '''
        nn.Parameter
        首先可把这个函数理解为类型转换函数，
        将一个不可训练的类型Tensor转换成可以训练的类型parameter
        并将这个parameter绑定到这个module里面(net.parameter()中就有这个绑定的parameter，
        所以在参数优化的时候可以进行优化的)
        '''
        self.W_R = nn.Parameter(torch.Tensor(self.n_relations, self.entity_dim, self.relation_dim))  #每个小r都是 k*d维的，扩展到所有r的集合R就是这样一个三维的
        nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))

        self.aggregator_layers = nn.ModuleList()
        for k in range(self.n_layers):
            self.aggregator_layers.append(Aggregator(self.conv_dim_list[k], self.conv_dim_list[k + 1], self.mess_dropout[k], self.aggregation_type))


    def att_score(self, edges):
        # Equation (4)
        # 注意获得尾实体和头实体的方式是不一样的。
        r_mul_t = torch.matmul(self.entity_user_embed(edges.src['id']), self.W_r)       # (n_edge, relation_dim)
        r_mul_h = torch.matmul(self.entity_user_embed(edges.dst['id']), self.W_r)       # (n_edge, relation_dim)
        r_embed = self.relation_embed(edges.data['type'])                               # (1, relation_dim)
        
        # torch.bmm两个三维矩阵相乘
        att = torch.bmm(r_mul_t.unsqueeze(1), torch.tanh(r_mul_h + r_embed).unsqueeze(2)).squeeze(-1)   # (n_edge, 1)
        return {'att': att}


    def compute_attention(self, g):
        g = g.local_var() 
        for i in range(self.n_relations):
            #filter_edges：Return the IDs of the edges with the given edge type that satisfy the given predicate.
            edge_idxs = g.filter_edges(lambda edge: edge.data['type'] == i)  #找到每种关系的边的id
            self.W_r = self.W_R[i]  
            
            # Update the features of the specified edges by the provided function(self.att_score).
            g.apply_edges(self.att_score, edge_idxs)

        # Equation (5)
        g.edata['att'] = edge_softmax_fix(g, g.edata.pop('att'))
        return g.edata.pop('att')

    
    def forward(self, mode, *input):
        if mode == 'calc_att':
            return self.compute_attention(*input)

    
