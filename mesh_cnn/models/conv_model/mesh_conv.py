import torch
from torch import nn
from torch.nn.functional import pad


class MeshConv(nn.Module):

    def __init__(self, in_channels, out_channels, k=5, bias=True):
        super(MeshConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, k),
            bias=bias,
        )
        self.k = k

    def __call__(self, edge_f, mesh):
        return self.forward(edge_f, mesh)

    def forward(self, x, mesh):
        x = x.squeeze(-1)
        G = torch.cat([self.pad_graph_mat(i, x.shape[2], x.device) for i in mesh], 0)
        G = self.create_graph_mat(x, G)
        x = self.conv(G)
        return x

    def flatten_graph_mat_idx(self, ne_idx):
        (b, ne, nn) = ne_idx.shape
        ne += 1
        batch_n = torch.floor(
            torch.arange(b * ne, device=ne_idx.device).float() / ne
        ).view(b, ne)
        norm_mat = batch_n * ne
        norm_mat = norm_mat.view(b, ne, 1)
        norm_mat = norm_mat.repeat(1, 1, nn)
        ne_idx = ne_idx.float() + norm_mat[:, 1:, :]
        return ne_idx

    def create_graph_mat(self, x, ne_idx):
        ne_idxshape = ne_idx.shape

        padding = torch.zeros(
            (x.shape[0], x.shape[1], 1), requires_grad=True, device=x.device
        )

        x = torch.cat((padding, x), dim=2)
        ne_idx = ne_idx + 1

        ne_idx_flat = self.flatten_graph_mat_idx(ne_idx)
        ne_idx_flat = ne_idx_flat.view(-1).long()

        odim = x.shape
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(odim[0] * odim[2], odim[1])

        f = torch.index_select(x, dim=0, index=ne_idx_flat)
        f = f.view(ne_idxshape[0], ne_idxshape[1], ne_idxshape[2], -1)
        f = f.permute(0, 3, 1, 2)

        edges = [f[:, :, :, _] for _ in range(5)]
        p = edges[1] + edges[3]
        q = edges[2] + edges[4]
        p1 = torch.abs(edges[1] - edges[3])
        q1 = torch.abs(edges[2] - edges[4])
        f = torch.stack([edges[0], p, q, p1, q1], dim=3)
        return f

    def pad_graph_mat(self, m, max_edges, device):
        padded_graph_mat = torch.tensor(m.graph_mat_edges, device=device).float()
        padded_graph_mat = padded_graph_mat.requires_grad_()
        padded_graph_mat = torch.cat(
            (
                torch.arange(m.edges_count, device=device).float().unsqueeze(1),
                padded_graph_mat,
            ),
            dim=1,
        )

        padded_graph_mat = pad(
            padded_graph_mat, (0, 0, 0, max_edges - m.edges_count), "constant", 0
        )
        padded_graph_mat = padded_graph_mat.unsqueeze(0)
        return padded_graph_mat
