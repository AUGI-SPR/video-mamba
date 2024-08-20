import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pickle
import copy
import numpy as np
import math
import shutil
from modeling.blocks import MaskMambaBlock, MaskMambaBlock_DBM

from eval import segment_bars_with_confidence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def exponential_descrease(idx_decoder, p=3):
    return math.exp(-p * idx_decoder)


class AttentionHelper(nn.Module):
    def __init__(self, args):
        super(AttentionHelper, self).__init__()
        self.args = args
        self.softmax = nn.Softmax(dim=-1)

    def scalar_dot_att(self, proj_query, proj_key, proj_val, padding_mask):
        """
        scalar dot attention.
        :param proj_query: shape of (B, C, L) => (Batch_Size, Feature_Dimension, Length)
        :param proj_key: shape of (B, C, L)
        :param proj_val: shape of (B, C, L)
        :param padding_mask: shape of (B, C, L)
        :return: attention value of shape (B, C, L)
        """
        m, c1, l1 = proj_query.shape
        m, c2, l2 = proj_key.shape

        assert c1 == c2

        energy = torch.bmm(
            proj_query.permute(0, 2, 1), proj_key
        )  # out of shape (B, L1, L2)
        attention = energy / np.sqrt(c1)
        attention = attention + torch.log(
            padding_mask + 1e-6
        )  # mask the zero paddings. log(1e-6) for zero paddings
        attention = self.softmax(attention)
        # print("attention shape: ", attention.shape)
        # print("proj_val shape: ", proj_val.shape)
        # print("attention shape: ", attention.shape)
        # print("padding_mask shape: ", padding_mask.shape)
        attention = attention * padding_mask
        # print("attention * padding_mask shape: ", attention.shape)
        attention = attention.permute(0, 2, 1)
        out = torch.bmm(proj_val, attention)
        return out, attention


class AttLayer(nn.Module):
    def __init__(
        self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type, args
    ):  # r1 = r2
        super(AttLayer, self).__init__()

        self.query_conv = nn.Conv1d(
            in_channels=q_dim, out_channels=q_dim // r1, kernel_size=1
        )
        self.key_conv = nn.Conv1d(
            in_channels=k_dim, out_channels=k_dim // r2, kernel_size=1
        )
        self.value_conv = nn.Conv1d(
            in_channels=v_dim, out_channels=v_dim // r3, kernel_size=1
        )

        self.conv_out = nn.Conv1d(
            in_channels=v_dim // r3, out_channels=v_dim, kernel_size=1
        )

        self.bl = bl
        self.stage = stage
        self.att_type = att_type
        self.args = args
        assert self.att_type in ["normal_att", "block_att", "sliding_att", "causal_att"]
        assert self.stage in ["encoder", "decoder"]

        self.att_helper = AttentionHelper(args)
        self.window_mask = self.construct_window_mask()

    def construct_window_mask(self):
        """
        construct window mask of shape (1, l, l + l//2 + l//2), used for sliding window self attention
        """
        window_mask = torch.zeros((1, self.bl, self.bl + 2 * (self.bl // 2)))
        for i in range(self.bl):
            window_mask[:, :, i : i + self.bl] = 1
        return window_mask.to(device)

    def forward(self, x1, x2, mask):
        # x1 from the encoder
        # x2 from the decoder

        query = self.query_conv(x1)
        key = self.key_conv(x1)

        if self.stage == "decoder":
            assert x2 is not None
            value = self.value_conv(x2)
        else:
            value = self.value_conv(x1)

        if self.att_type == "normal_att":
            return self._normal_self_att(query, key, value, mask)
        elif self.att_type == "block_att":
            return self._block_wise_self_att(query, key, value, mask)
        elif self.att_type == "sliding_att":
            return self._sliding_window_self_att(query, key, value, mask)
        elif self.att_type == "causal_att":
            return self._causal_self_att(query, key, value, mask)

    def _normal_self_att(self, q, k, v, mask):
        m_batchsize, c1, L = q.size()
        _, c2, L = k.size()
        _, c3, L = v.size()
        padding_mask = torch.ones((m_batchsize, 1, L)).to(device) * mask[:, 0:1, :]
        output, attentions = self.att_helper.scalar_dot_att(q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]

    def _block_wise_self_att(self, q, k, v, mask):
        m_batchsize, c1, L = q.size()
        _, c2, L = k.size()
        _, c3, L = v.size()

        nb = L // self.bl
        if L % self.bl != 0:
            q = torch.cat(
                [q, torch.zeros((m_batchsize, c1, self.bl - L % self.bl)).to(device)],
                dim=-1,
            )
            k = torch.cat(
                [k, torch.zeros((m_batchsize, c2, self.bl - L % self.bl)).to(device)],
                dim=-1,
            )
            v = torch.cat(
                [v, torch.zeros((m_batchsize, c3, self.bl - L % self.bl)).to(device)],
                dim=-1,
            )
            nb += 1

        padding_mask = torch.cat(
            [
                torch.ones((m_batchsize, 1, L)).to(device) * mask[:, 0:1, :],
                torch.zeros((m_batchsize, 1, self.bl * nb - L)).to(device),
            ],
            dim=-1,
        )

        q = (
            q.reshape(m_batchsize, c1, nb, self.bl)
            .permute(0, 2, 1, 3)
            .reshape(m_batchsize * nb, c1, self.bl)
        )
        padding_mask = (
            padding_mask.reshape(m_batchsize, 1, nb, self.bl)
            .permute(0, 2, 1, 3)
            .reshape(m_batchsize * nb, 1, self.bl)
        )
        k = (
            k.reshape(m_batchsize, c2, nb, self.bl)
            .permute(0, 2, 1, 3)
            .reshape(m_batchsize * nb, c2, self.bl)
        )
        v = (
            v.reshape(m_batchsize, c3, nb, self.bl)
            .permute(0, 2, 1, 3)
            .reshape(m_batchsize * nb, c3, self.bl)
        )

        output, attentions = self.att_helper.scalar_dot_att(q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))

        output = (
            output.reshape(m_batchsize, nb, c3, self.bl)
            .permute(0, 2, 1, 3)
            .reshape(m_batchsize, c3, nb * self.bl)
        )
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]

    def _sliding_window_self_att(self, q, k, v, mask):
        m_batchsize, c1, L = q.size()
        _, c2, _ = k.size()
        _, c3, _ = v.size()

        assert m_batchsize == 1  # currently, we only accept input with batch size 1
        # padding zeros for the last segment
        nb = L // self.bl
        if L % self.bl != 0:
            q = torch.cat(
                [q, torch.zeros((m_batchsize, c1, self.bl - L % self.bl)).to(device)],
                dim=-1,
            )
            k = torch.cat(
                [k, torch.zeros((m_batchsize, c2, self.bl - L % self.bl)).to(device)],
                dim=-1,
            )
            v = torch.cat(
                [v, torch.zeros((m_batchsize, c3, self.bl - L % self.bl)).to(device)],
                dim=-1,
            )
            nb += 1
        padding_mask = torch.cat(
            [
                torch.ones((m_batchsize, 1, L)).to(device) * mask[:, 0:1, :],
                torch.zeros((m_batchsize, 1, self.bl * nb - L)).to(device),
            ],
            dim=-1,
        )

        # sliding window approach, by splitting query_proj and key_proj into shape (c1, l) x (c1, 2l)
        # sliding window for query_proj: reshape
        q = (
            q.reshape(m_batchsize, c1, nb, self.bl)
            .permute(0, 2, 1, 3)
            .reshape(m_batchsize * nb, c1, self.bl)
        )

        # sliding window approach for key_proj
        # 1. add paddings at the start and end
        k = torch.cat(
            [
                torch.zeros(m_batchsize, c2, self.bl // 2).to(device),
                k,
                torch.zeros(m_batchsize, c2, self.bl // 2).to(device),
            ],
            dim=-1,
        )
        v = torch.cat(
            [
                torch.zeros(m_batchsize, c3, self.bl // 2).to(device),
                v,
                torch.zeros(m_batchsize, c3, self.bl // 2).to(device),
            ],
            dim=-1,
        )
        padding_mask = torch.cat(
            [
                torch.zeros(m_batchsize, 1, self.bl // 2).to(device),
                padding_mask,
                torch.zeros(m_batchsize, 1, self.bl // 2).to(device),
            ],
            dim=-1,
        )

        # 2. reshape key_proj of shape (m_batchsize*nb, c1, 2*self.bl)
        k = torch.cat(
            [
                k[:, :, i * self.bl : (i + 1) * self.bl + (self.bl // 2) * 2]
                for i in range(nb)
            ],
            dim=0,
        )  # special case when self.bl = 1
        v = torch.cat(
            [
                v[:, :, i * self.bl : (i + 1) * self.bl + (self.bl // 2) * 2]
                for i in range(nb)
            ],
            dim=0,
        )
        # 3. construct window mask of shape (1, l, 2l), and use it to generate final mask
        padding_mask = torch.cat(
            [
                padding_mask[:, :, i * self.bl : (i + 1) * self.bl + (self.bl // 2) * 2]
                for i in range(nb)
            ],
            dim=0,
        )  # of shape (m*nb, 1, 2l)
        final_mask = self.window_mask.repeat(m_batchsize * nb, 1, 1) * padding_mask
        output, attention = self.att_helper.scalar_dot_att(q, k, v, final_mask)
        output = self.conv_out(F.relu(output))
        output = (
            output.reshape(m_batchsize, nb, -1, self.bl)
            .permute(0, 2, 1, 3)
            .reshape(m_batchsize, -1, nb * self.bl)
        )
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]

    def _causal_self_att(self, q, k, v, mask):
        m_batchsize, c1, L = q.size()
        _, c2, L = k.size()
        _, c3, L = v.size()

        nb = L // self.bl
        if L % self.bl != 0:
            q = torch.cat(
                [q, torch.zeros((m_batchsize, c1, self.bl - L % self.bl)).to(device)],
                dim=-1,
            )
            k = torch.cat(
                [k, torch.zeros((m_batchsize, c2, self.bl - L % self.bl)).to(device)],
                dim=-1,
            )
            v = torch.cat(
                [v, torch.zeros((m_batchsize, c3, self.bl - L % self.bl)).to(device)],
                dim=-1,
            )
            nb += 1

        padding_mask = torch.cat(
            [
                torch.ones((m_batchsize, 1, L)).to(device) * mask[:, 0:1, :],
                torch.zeros((m_batchsize, 1, self.bl * nb - L)).to(device),
            ],
            dim=-1,
        )

        q = (
            q.reshape(m_batchsize, c1, nb, self.bl)
            .permute(0, 2, 1, 3)
            .reshape(m_batchsize * nb, c1, self.bl)
        )
        padding_mask = (
            padding_mask.reshape(m_batchsize, 1, nb, self.bl)
            .permute(0, 2, 1, 3)
            .reshape(m_batchsize * nb, 1, self.bl)
        )
        k = (
            k.reshape(m_batchsize, c2, nb, self.bl)
            .permute(0, 2, 1, 3)
            .reshape(m_batchsize * nb, c2, self.bl)
        )
        v = (
            v.reshape(m_batchsize, c3, nb, self.bl)
            .permute(0, 2, 1, 3)
            .reshape(m_batchsize * nb, c3, self.bl)
        )
        causal_mask = torch.tril(torch.ones((self.bl, self.bl))).unsqueeze(0).to(device)
        padding_mask = padding_mask * causal_mask
        output, attentions = self.att_helper.scalar_dot_att(q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))
        output = (
            output.reshape(m_batchsize, nb, -1, self.bl)
            .permute(0, 2, 1, 3)
            .reshape(m_batchsize, -1, nb * self.bl)
        )
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]


class MultiHeadAttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type, num_head):
        super(MultiHeadAttLayer, self).__init__()
        #         assert v_dim % num_head == 0
        self.conv_out = nn.Conv1d(v_dim * num_head, v_dim, 1)
        self.layers = nn.ModuleList(
            [
                copy.deepcopy(
                    AttLayer(q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type)
                )
                for i in range(num_head)
            ]
        )
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x1, x2, mask):
        out = torch.cat([layer(x1, x2, mask) for layer in self.layers], dim=1)
        out = self.conv_out(self.dropout(out))
        return out


class ConvFeedForward(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(ConvFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, 3, padding=dilation, dilation=dilation
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class FCFeedForward(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),  # conv1d equals fc
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(out_channels, out_channels, 1),
        )

    def forward(self, x):
        return self.layer(x)


class AttModule(nn.Module):
    def __init__(
        self, dilation, in_channels, out_channels, r1, r2, att_type, stage, alpha, args
    ):
        super(AttModule, self).__init__()
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels)
        self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        self.att_layer = AttLayer(
            in_channels,
            in_channels,
            out_channels,
            r1,
            r1,
            r2,
            dilation,
            att_type=att_type,
            stage=stage,
            args=args,
        )  # dilation
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.alpha = alpha

    def forward(self, x, f, mask):
        out = self.feed_forward(x)
        out = self.alpha * self.att_layer(self.instance_norm(out), f, mask) + out
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class AttModule_mamba(nn.Module):
    def __init__(
        self,
        dilation,
        in_channels,
        out_channels,
        r1,
        r2,
        att_type,
        stage,
        alpha,
        drop_path_rate=0.3,
        args=None,
    ):
        super(AttModule_mamba, self).__init__()
        self.args = args
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels)
        self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        self.att_layer = MaskMambaBlock(
            in_channels, drop_path_rate=drop_path_rate
        )  # dilation
        # self.att_layer = MaskMambaBlock_DBM(in_channels, drop_path_rate=drop_path_rate) # dilation
        self.conv_1x1 = nn.Conv1d(in_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.alpha = alpha

    def forward(self, x, f, mask):
        m_batchsize, c1, L = x.size()
        padding_mask = torch.ones((m_batchsize, 1, L)).to(device) * mask[:, 0:1, :]
        out = self.feed_forward(x)
        out = self.alpha * self.att_layer(self.instance_norm(out), padding_mask) + out
        # out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).permute(0, 2, 1)  # of shape (1, d_model, l)
        self.pe = nn.Parameter(pe, requires_grad=True)

    #         self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :, 0 : x.shape[2]]


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers,
        r1,
        r2,
        num_f_maps,
        input_dim,
        num_classes,
        channel_masking_rate,
        att_type,
        alpha,
        mamba=False,
        drop_path_rate=0.3,
        args=None,
    ):
        super(Encoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)  # fc layer
        if not mamba:
            self.layers = nn.ModuleList(
                [
                    AttModule(
                        2**i,
                        num_f_maps,
                        num_f_maps,
                        r1,
                        r2,
                        att_type,
                        "encoder",
                        alpha,
                        args,
                    )
                    for i in range(num_layers)  # 2**i
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    AttModule_mamba(
                        2**i,
                        num_f_maps,
                        num_f_maps,
                        r1,
                        r2,
                        att_type,
                        "encoder",
                        alpha,
                        drop_path_rate=drop_path_rate,
                        args=args,
                    )
                    for i in range(num_layers)  # 2**i
                ]
            )

        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate

    def forward(self, x, mask):
        """
        :param x: (N, C, L)
        :param mask:
        :return:
        """

        if self.channel_masking_rate > 0:
            x = x.unsqueeze(2)
            x = self.dropout(x)
            x = x.squeeze(2)

        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, None, mask)

        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature


class Decoder(nn.Module):
    def __init__(
        self,
        num_layers,
        r1,
        r2,
        num_f_maps,
        input_dim,
        num_classes,
        att_type,
        alpha,
        mamba=False,
        drop_path_rate=0.3,
        args=None,
    ):
        super(
            Decoder, self
        ).__init__()  #         self.position_en = PositionalEncoding(d_model=num_f_maps)
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        if not mamba:
            self.layers = nn.ModuleList(
                [
                    AttModule(
                        2**i,
                        num_f_maps,
                        num_f_maps,
                        r1,
                        r2,
                        att_type,
                        "decoder",
                        alpha,
                        args=args,
                    )
                    for i in range(num_layers)  # 2 ** i
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    AttModule_mamba(
                        2**i,
                        num_f_maps,
                        num_f_maps,
                        r1,
                        r2,
                        att_type,
                        "decoder",
                        alpha,
                        drop_path_rate=drop_path_rate,
                        args=args,
                    )
                    for i in range(num_layers)  # 2 ** i
                ]
            )
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, fencoder, mask):

        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, fencoder, mask)

        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature


class MyTransformer(nn.Module):
    def __init__(
        self,
        num_decoders,
        num_layers,
        r1,
        r2,
        num_f_maps,
        input_dim,
        num_classes,
        channel_masking_rate,
        drop_path_rate=0.3,
        encoder_only=False,
        args=None,
    ):
        super(MyTransformer, self).__init__()
        self.encoder_only = encoder_only
        self.encoder = Encoder(
            num_layers,
            r1,
            r2,
            num_f_maps,
            input_dim,
            num_classes,
            channel_masking_rate,
            att_type="causal_att" if args.causal else "sliding_att",
            alpha=1,
            args=args,
        )
        if encoder_only:
            self.decoders = nn.ModuleList(
                [
                    copy.deepcopy(
                        Encoder(
                            num_layers,
                            r1,
                            r2,
                            num_f_maps,
                            num_classes,
                            num_classes,
                            channel_masking_rate,
                            att_type="causal_att" if args.causal else "sliding_att",
                            alpha=exponential_descrease(s),
                            args=args,
                        )
                    )
                    for s in range(num_decoders)
                ]
            )  # num_decoders
        else:
            self.decoders = nn.ModuleList(
                [
                    copy.deepcopy(
                        Decoder(
                            num_layers,
                            r1,
                            r2,
                            num_f_maps,
                            num_classes,
                            num_classes,
                            att_type="sliding_att",
                            alpha=exponential_descrease(s),
                            args=args,
                        )
                    )
                    for s in range(num_decoders)
                ]
            )  # num_decoders

    def forward(self, x, mask):
        out, feature = self.encoder(x, mask)
        outputs = out.unsqueeze(0)

        for decoder in self.decoders:
            if not self.encoder_only:
                out, feature = decoder(
                    F.softmax(out, dim=1) * mask[:, 0:1, :],
                    feature * mask[:, 0:1, :],
                    mask,
                )
            else:
                out, feature = decoder(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        return outputs


class MaTransformer(nn.Module):
    def __init__(
        self,
        num_decoders,
        num_layers,
        r1,
        r2,
        num_f_maps,
        input_dim,
        num_classes,
        channel_masking_rate,
        drop_path_rate=0.3,
        args=None,
    ):
        super(MaTransformer, self).__init__()
        self.encoder = Encoder(
            num_layers,
            r1,
            r2,
            num_f_maps,
            input_dim,
            num_classes,
            channel_masking_rate,
            att_type="sliding_att",
            alpha=1,
            mamba=True,
            drop_path_rate=drop_path_rate,
            args=args,
        )
        self.decoders = nn.ModuleList(
            [
                copy.deepcopy(
                    Decoder(
                        num_layers,
                        r1,
                        r2,
                        num_f_maps,
                        num_classes,
                        num_classes,
                        att_type="sliding_att",
                        alpha=exponential_descrease(s),
                        drop_path_rate=drop_path_rate,
                        args=args,
                    )
                )
                for s in range(num_decoders)
            ]
        )  # num_decoders

    def forward(self, x, mask):
        out, feature = self.encoder(x, mask)
        outputs = out.unsqueeze(0)

        for decoder in self.decoders:
            out, feature = decoder(
                F.softmax(out, dim=1) * mask[:, 0:1, :], feature * mask[:, 0:1, :], mask
            )
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        return outputs


class Trainer:
    def __init__(
        self,
        num_layers,
        r1,
        r2,
        num_f_maps,
        input_dim,
        num_classes,
        channel_masking_rate,
        mamba=False,
        drop_path_rate=0.3,
        args=None,
    ):
        if not mamba:
            self.model = MyTransformer(
                3,
                num_layers,
                r1,
                r2,
                num_f_maps,
                input_dim,
                num_classes,
                channel_masking_rate,
                encoder_only=args.encoder_only,
                args=args,
            )
        else:
            self.model = MaTransformer(
                3,
                num_layers,
                r1,
                r2,
                num_f_maps,
                input_dim,
                num_classes,
                channel_masking_rate,
                drop_path_rate=drop_path_rate,
                args=args,
            )
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.args = args
        # print("Model Size: ", sum(p.numel() for p in self.model.parameters()))
        self.mse = nn.MSELoss(reduction="none")
        self.num_classes = num_classes

    def train(
        self,
        save_dir,
        batch_gen,
        num_epochs,
        batch_size,
        learning_rate,
        batch_gen_tst=None,
        patience=10,  # Adding a patience parameter for early stopping
    ):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        # print("LR:{}".format(learning_rate))

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )

        best_acc = 0
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0

            while batch_gen.has_next():
                batch_input, batch_target, mask, vids = batch_gen.next_batch(
                    batch_size, False
                )
                batch_input, batch_target, mask = (
                    batch_input.to(device),
                    batch_target.to(device),
                    mask.to(device),
                )
                optimizer.zero_grad()
                ps = self.model(batch_input, mask)

                loss = 0
                for p in ps:
                    loss += self.ce(
                        p.transpose(2, 1).contiguous().view(-1, self.num_classes),
                        batch_target.view(-1),
                    )
                    loss += 0.15 * torch.mean(
                        torch.clamp(
                            self.mse(
                                F.log_softmax(p[:, :, 1:], dim=1),
                                F.log_softmax(p.detach()[:, :, :-1], dim=1),
                            ),
                            min=0,
                            max=16,
                        )
                        * mask[:, :, 1:]
                    )

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(ps.data[-1], 1)
                correct += (
                    ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1))
                    .sum()
                    .item()
                )
                total += torch.sum(mask[:, 0, :]).item()

            scheduler.step(epoch_loss)
            batch_gen.reset()
            epoch_acc = float(correct) / total
            print(
                "[epoch %d]: epoch loss = %f,   acc = %f"
                % (
                    epoch + 1,
                    epoch_loss / len(batch_gen.list_of_examples),
                    epoch_acc,
                )
            )

            # Calculate test accuracy if test generator is provided
            if batch_gen_tst is not None:
                test_acc = self.test(batch_gen_tst, epoch)
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_epoch = epoch + 1

                    # Remove all previous best models
                    for filename in os.listdir(save_dir):
                        if filename.startswith("best_"):
                            file_path = os.path.join(save_dir, filename)
                            try:
                                if os.path.isfile(file_path) or os.path.islink(
                                    file_path
                                ):
                                    os.unlink(file_path)
                                elif os.path.isdir(file_path):
                                    shutil.rmtree(file_path)
                            except Exception as e:
                                print(f"Failed to delete {file_path}. Reason: {e}")

                    # Save the new best model
                    torch.save(
                        self.model.state_dict(),
                        save_dir + f"/best_{best_epoch}.model",
                    )
                    torch.save(
                        optimizer.state_dict(),
                        save_dir + f"/best_{best_epoch}.opt",
                    )
                    print(
                        f"Best model saved at epoch {best_epoch} with test accuracy {best_acc:.4f}"
                    )
                    epochs_without_improvement = 0  # Reset patience counter

                else:
                    epochs_without_improvement += 1
                    print(f"No improvement for {epochs_without_improvement} epochs.")

            # Early stopping condition
            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping triggered after {epochs_without_improvement} epochs without improvement."
                )
                break

            # Save current epoch's model
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            torch.save(
                self.model.state_dict(),
                save_dir + "/epoch-" + str(epoch + 1) + ".model",
            )
            torch.save(
                optimizer.state_dict(),
                save_dir + "/epoch-" + str(epoch + 1) + ".opt",
            )

        print(
            f"Training complete. Best model saved at epoch {best_epoch} with test accuracy {best_acc:.4f}"
        )

    def test(self, batch_gen_tst, epoch):
        self.model.eval()
        correct = 0
        total = 0
        if_warp = False  # When testing, always false
        with torch.no_grad():
            while batch_gen_tst.has_next():
                batch_input, batch_target, mask, vids = batch_gen_tst.next_batch(
                    1, if_warp
                )
                batch_input, batch_target, mask = (
                    batch_input.to(device),
                    batch_target.to(device),
                    mask.to(device),
                )
                p = self.model(batch_input, mask)
                _, predicted = torch.max(p.data[-1], 1)
                correct += (
                    ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1))
                    .sum()
                    .item()
                )
                total += torch.sum(mask[:, 0, :]).item()

        acc = float(correct) / total
        print("---[epoch %d]---: tst acc = %f" % (epoch + 1, acc))

        self.model.train()
        batch_gen_tst.reset()

        return acc

    def predict(
        self,
        model_dir,
        result_dir,
        batch_gen_tst,
        epoch,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        best_model_path = self._find_best_model(model_dir)
        if best_model_path:
            print(f"Loading best model from: {best_model_path}")
            self.model.load_state_dict(torch.load(best_model_path))
            self.model.to(device)
        else:
            print("No best model found.")
            return

        self.model.eval()
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            while batch_gen_tst.has_next():
                batch_input, batch_target, mask, vids = batch_gen_tst.next_batch(
                    1, if_warp=False
                )
                batch_input, batch_target, mask = (
                    batch_input.to(device),
                    batch_target.to(device),
                    mask.to(device),
                )

                # Get predictions from the model
                p = self.model(batch_input, mask)
                _, predicted = torch.max(p.data[-1], 1)

                # Calculate accuracy
                correct_predictions += (
                    ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1))
                    .sum()
                    .item()
                )
                total_predictions += torch.sum(mask[:, 0, :]).item()

                # Generate phase recognition plot
                vid = vids[0]  # Assuming vids is a list with a single video ID
                save_path = f"{result_dir}/{vid}_epoch{epoch}_plot.png"
                self.plot_phase_recognition(
                    save_path, predicted.cpu().numpy(), batch_target.cpu().numpy()
                )

        # Calculate overall accuracy
        accuracy = correct_predictions / total_predictions
        print(f"Prediction complete. Accuracy: {accuracy * 100:.2f}%")

    def _find_best_model(self, model_dir):
        # Find the latest best model file in the directory
        best_models = [
            f
            for f in os.listdir(model_dir)
            if f.startswith("best_") and f.endswith(".model")
        ]
        if not best_models:
            return None
        best_model_path = max(
            [os.path.join(model_dir, f) for f in best_models], key=os.path.getctime
        )
        return best_model_path

    def plot_phase_recognition(self, save_path, predicted, batch_target):
        # Plot the phase recognition results
        segment_bars_with_confidence(save_path, None, batch_target, predicted)


if __name__ == "__main__":
    num_layers = 10
    num_f_maps = 64
    features_dim = 768
    bz = 1
    num_classes = 7
    channel_mask_rate = 0.3
    x = torch.rand(1, features_dim, 100).cuda()
    mask = torch.ones(1, num_classes, 100).cuda()

    model = MyTransformer(
        3,
        num_layers,
        2,
        2,
        num_f_maps,
        features_dim,
        num_classes,
        0.1,
    ).cuda()
    out2 = model(x, mask)
    print(out2.shape)

    model = MaTransformer(
        3, num_layers, 2, 2, num_f_maps, features_dim, num_classes, 0.1
    ).cuda()
    out = model(x, mask)
    print(out.shape)
    import pdb

    pdb.set_trace()
    x = 0
