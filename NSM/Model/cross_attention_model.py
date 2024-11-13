import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadCrossAttention(nn.Module):
    def __init__(self,x_emb,y_emb):
        super(MultiHeadCrossAttention, self).__init__()
        self.batch_size,self.num,self.in_dim=y_emb.shape
        # print(x_emb)
        self.in_q_dim=x_emb.shape[2]
        self.x=x_emb
        self.y=y_emb
        self.out_dim = 128
        self.head_num=8
        self._norm_fact = 1 / math.sqrt(self.out_dim // self.head_num)
        # qkv输出的维度自己指定，但必须一样（所以这里就没有分开指定，直接out_dim）；且输出维度能整除num_head
        assert self.out_dim % self.head_num == 0, "dim_k and dim_v must be multiple of num_heads"

        # 定义查询、键、值三个线性变换
        self.query = nn.Linear(self.in_q_dim, self.out_dim, bias=False, dtype=torch.float32)  # 变化 300->128
        self.key = nn.Linear(self.in_dim, self.out_dim, bias=False, dtype=torch.float32)  # 50->128
        self.value = nn.Linear(self.in_dim, self.out_dim, bias=False, dtype=torch.float32) #50->128

    def forward(self):
        dim= self.out_dim // self.head_num  # dim_k of each head

        x_ = self.query(self.x).reshape(self.batch_size,self.num,self.head_num,dim).transpose(1,2)  # 查询矩阵
        y_ = self.key(self.y).reshape(self.batch_size,self.num,self.head_num,dim).transpose(1,2)  # 键值矩阵
        # 计算注意力分数
        attn_scores = torch.matmul(x_, y_.transpose(-2, -1)) * self._norm_fact  # 计算注意力分数，注意力分数矩阵的大小为 batch_size x num_queries x num_keys x num_keys
        attn_weights = F.softmax(attn_scores, dim=-1)  # 对注意力分数进行 softmax 归一化
        # 计算加权和
        V = self.value(self.y).reshape(self.batch_size,self.num,self.head_num,dim).transpose(1,2)  # 通过值变换得到值矩阵 V

        output=torch.matmul(attn_weights,V).reshape(self.batch_size,self.num,self.out_dim)
        # output = torch.bmm(attn_weights, V)  # 计算加权和，output 的大小为 batch_size x num_queries x num_keys x out_dim
        att_output = torch.softmax(output, dim=-1)  # 自己增加了一层softmax，归一化为一个概率分布向量，方便后续kl损失使用。
        return att_output