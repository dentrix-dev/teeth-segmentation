from torch import nn
from models.conv_model.upDownConv import UpConv, DownConv


class UNet(nn.Module):

    def __init__(self, pools, conv, deconv, blocks=0, transfer_data=True):
        super(UNet, self).__init__()
        self.transfer_data = transfer_data
        self.encoder = Encoder(pools, conv, blocks=blocks)
        unrolls = (pools[:-1].copy())[::-1]
        self.decoder = Decoder(
            unrolls, deconv, blocks=blocks, transfer_data=transfer_data
        )

    def forward(self, x, meshes):
        feautres_emb, unpooled = self.encoder((x, meshes))
        feautres_emb = self.decoder((feautres_emb, meshes), unpooled)
        return feautres_emb

    def __call__(self, x, meshes):
        return self.forward(x, meshes)


class Encoder(nn.Module):
    def __init__(self, pools, convs, fcs=None, blocks=0):
        super(Encoder, self).__init__()
        self.relu = nn.ReLU()
        self.convs = []

        for i in range(len(convs) - 1):
            pool = pools[i + 1] if i + 1 < len(pools) else 0
            self.convs.append(
                DownConv(convs[i], convs[i + 1], blocks=blocks, pool=pool)
            )

        self.convs = nn.ModuleList(self.convs)
        init_params(self)

    def forward(self, x):
        feautres_emb, meshes = x
        encoder_outs = []
        for conv in self.convs:
            feautres_emb, unpooled = conv((feautres_emb, meshes))
            encoder_outs.append(unpooled)

        return feautres_emb, encoder_outs

    def __call__(self, x):
        return self.forward(x)


class Decoder(nn.Module):
    def __init__(self, unrolls, convs, blocks=0, batch_norm=True, transfer_data=True):
        super(Decoder, self).__init__()
        self.deconv = []
        for i in range(len(convs) - 2):
            unroll = unrolls[i] if i + 1 < len(unrolls) else 0
            self.deconv.append(
                UpConv(
                    convs[i],
                    convs[i + 1],
                    blocks=blocks,
                    unroll=unroll,
                    batch_norm=batch_norm,
                    transfer_data=transfer_data,
                )
            )
        self.final_conv = UpConv(
            convs[-2],
            convs[-1],
            blocks=blocks,
            unroll=False,
            batch_norm=batch_norm,
            transfer_data=False,
        )
        self.deconv = nn.ModuleList(self.deconv)
        init_params(self)

    def forward(self, x, encoder_outs=None):
        feautres_emb, meshes = x
        for i, up_conv in enumerate(self.deconv):
            unpooled = None
            if encoder_outs is not None:
                unpooled = encoder_outs[-(i + 2)]
            feautres_emb = up_conv((feautres_emb, meshes), unpooled)
        feautres_emb = self.final_conv((feautres_emb, meshes))
        return feautres_emb

    def __call__(self, x, encoder_outs=None):
        return self.forward(x, encoder_outs)


def init_params(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
