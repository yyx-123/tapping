import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, avg_pool, SAGEConv, ChebConv, GATConv, ARMAConv, SuperGATConv
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import BatchNorm


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout, usePooling=True, ratio=0.8):
        super(GCN, self).__init__()

        self.gc1 = GCNConv(nfeat, nhid)
        self.pool1 = TopKPooling(in_channels=nhid, ratio=ratio)
        self.gc2 = GCNConv(nhid, nout)
        self.pool2 = TopKPooling(in_channels=nout, ratio=ratio)
        self.fc1 = nn.Linear(nout, 3)
        self.dropout = dropout
        self.usePooling = usePooling
    def forward(self, G_):
        G = G_.clone()
        G.x = F.relu(self.gc1(G.x, G.edge_index))
        G.x, G.edge_index, G.edge_attr, G.batch, _, _ = self.pool1(G.x, G.edge_index, batch=G.batch)
        G.x = F.dropout(G.x, self.dropout, training=self.training)
        G.x = F.relu(self.gc2(G.x, G.edge_index))
        G.x, G.edge_index, G.edge_attr, G.batch, _, _ = self.pool2(G.x, G.edge_index, batch=G.batch)
        G = avg_pool(G.batch, G)
        x = self.fc1(G.x)
        return F.log_softmax(x, dim=1)

class GraphSAGE(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout,usePooling=True, ratio=0.8):
        super(GraphSAGE, self).__init__()

        self.gc1 = SAGEConv(nfeat, nhid)
        self.pool1 = TopKPooling(in_channels=nhid, ratio=ratio)
        self.gc2 = SAGEConv(nhid, nout)
        self.pool2 = TopKPooling(in_channels=nout, ratio=ratio)
        self.fc1 = nn.Linear(nout, 3)
        self.dropout = dropout
        self.usePooling = usePooling
    def forward(self, G_):
        G = G_.clone()
        G.x = F.relu(self.gc1(G.x, G.edge_index))
        G.x, G.edge_index, G.edge_attr, G.batch, _, _ = self.pool1(G.x, G.edge_index, batch=G.batch)
        G.x = F.dropout(G.x, self.dropout, training=self.training)
        G.x = F.relu(self.gc2(G.x, G.edge_index))
        G.x, G.edge_index, G.edge_attr, G.batch, _, _ = self.pool2(G.x, G.edge_index, batch=G.batch)
        G = avg_pool(G.batch, G)
        x = self.fc1(G.x)
        return F.log_softmax(x, dim=1)

class ChebNet(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout, usePooling=True, ratio=0.8):
        super(ChebNet, self).__init__()

        self.gc1 = ChebConv(nfeat, nhid, K=2)
        self.pool1 = TopKPooling(in_channels=nhid, ratio=ratio)
        self.gc2 = ChebConv(nhid, nout, K=2)
        self.pool2 = TopKPooling(in_channels=nout, ratio=ratio)
        self.fc1 = nn.Linear(nout, 3)
        self.dropout = dropout
        self.usePooling = usePooling
    def forward(self, G_):
        G = G_.clone()
        G.x = F.relu(self.gc1(G.x, G.edge_index))
        G.x, G.edge_index, G.edge_attr, G.batch, _, _ = self.pool1(G.x, G.edge_index, batch=G.batch)
        G.x = F.dropout(G.x, self.dropout, training=self.training)
        G.x = F.relu(self.gc2(G.x, G.edge_index))
        G.x, G.edge_index, G.edge_attr, G.batch, _, _ = self.pool2(G.x, G.edge_index, batch=G.batch)
        G = avg_pool(G.batch, G)
        x = self.fc1(G.x)
        return F.log_softmax(x, dim=1)


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout, usePooling=True, ratio=0.8):
        super(GAT, self).__init__()

        self.gc1 = GATConv(nfeat, nhid)
        self.pool1 = TopKPooling(in_channels=nhid, ratio=ratio)
        self.gc2 = GATConv(nhid, nout)
        self.pool2 = TopKPooling(in_channels=nout, ratio=ratio)
        self.fc1 = nn.Linear(nout, 3)
        self.dropout = dropout
        self.usePooling = usePooling
    def forward(self, G_):
        G = G_.clone()
        G.x = F.relu(self.gc1(G.x, G.edge_index))
        G.x, G.edge_index, G.edge_attr, G.batch, _, _ = self.pool1(G.x, G.edge_index, batch=G.batch)
        G.x = F.dropout(G.x, self.dropout, training=self.training)
        G.x = F.relu(self.gc2(G.x, G.edge_index))
        G.x, G.edge_index, G.edge_attr, G.batch, _, _ = self.pool2(G.x, G.edge_index, batch=G.batch)
        G = avg_pool(G.batch, G)
        x = self.fc1(G.x)
        return F.log_softmax(x, dim=1)

class SuperGAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout, usePooling=True, ratio=0.8):
        super(SuperGAT, self).__init__()

        self.gc1 = SuperGATConv(nfeat, nhid)
        self.pool1 = TopKPooling(in_channels=nhid, ratio=ratio)
        self.gc2 = SuperGATConv(nhid, nout)
        self.pool2 = TopKPooling(in_channels=nout, ratio=ratio)
        self.fc1 = nn.Linear(nout, 3)
        self.dropout = dropout
        self.usePooling = usePooling

    def forward(self, G_):
        G = G_.clone()
        G.x = F.relu(self.gc1(G.x, G.edge_index))
        G.x, G.edge_index, G.edge_attr, G.batch, _, _ = self.pool1(G.x, G.edge_index, batch=G.batch)
        G.x = F.dropout(G.x, self.dropout, training=self.training)
        G.x = F.relu(self.gc2(G.x, G.edge_index))
        G.x, G.edge_index, G.edge_attr, G.batch, _, _ = self.pool2(G.x, G.edge_index, batch=G.batch)
        G = avg_pool(G.batch, G)
        x = self.fc1(G.x)
        return F.log_softmax(x, dim=1)

class ARMA(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout, usePooling=True, ratio=0.8):
        super(ARMA, self).__init__()

        self.gc1 = ARMAConv(nfeat, nhid)
        self.bn1 = BatchNorm(nhid)
        self.pool1 = TopKPooling(in_channels=nhid, ratio=ratio)
        self.gc2 = ARMAConv(nhid, nout)
        self.bn2 = BatchNorm(nout)
        self.pool2 = TopKPooling(in_channels=nout, ratio=ratio)
        self.fc1 = nn.Linear(nout, 3)
        self.dropout = dropout
        self.usePooling = usePooling
    def forward(self, G_):
        G = G_.clone()
        G.x = F.relu(self.gc1(G.x, G.edge_index))
        G.x, G.edge_index, G.edge_attr, G.batch, _, _ = self.pool1(G.x, G.edge_index, batch=G.batch)
        G.x = F.dropout(G.x, self.dropout, training=self.training)
        G.x = F.relu(self.gc2(G.x, G.edge_index))
        G.x, G.edge_index, G.edge_attr, G.batch, _, _ = self.pool2(G.x, G.edge_index, batch=G.batch)

        G = avg_pool(G.batch, G)

        x = self.fc1(G.x)
        return F.log_softmax(x, dim=1)

