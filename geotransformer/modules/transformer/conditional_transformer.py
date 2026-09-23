import torch.nn as nn

from geotransformer.modules.transformer.lrpe_transformer import LRPETransformerLayer
from geotransformer.modules.transformer.pe_transformer import PETransformerLayer
from geotransformer.modules.transformer.rpe_transformer import RPETransformerLayer
from geotransformer.modules.transformer.vanilla_transformer import TransformerLayer


def _check_block_type(block):
    if block not in ['self', 'cross']:
        raise ValueError('Unsupported block type "{}".'.format(block))


class VanillaConditionalTransformer(nn.Module):
    def __init__(self, blocks, d_model, num_heads, dropout=None, activation_fn='ReLU', return_attention_scores=False):
        super(VanillaConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores

    def forward(self, feats0, feats1, masks0=None, masks1=None):
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers[i](feats0, feats0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, memory_masks=masks1)
            else:
                feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1


class PEConditionalTransformer(nn.Module):
    def __init__(self, blocks, d_model, num_heads, dropout=None, activation_fn='ReLU', return_attention_scores=False):
        super(PEConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            if block == 'self':
                layers.append(PETransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
            else:
                layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores

    def forward(self, feats0, feats1, embeddings0, embeddings1, masks0=None, masks1=None):
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers[i](feats0, feats0, embeddings0, embeddings0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, embeddings1, embeddings1, memory_masks=masks1)
            else:
                feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1


class RPEConditionalTransformer(nn.Module):
    def __init__(
        self,
        blocks,
        d_model,
        num_heads,
        dropout=None,
        activation_fn='ReLU',
        return_attention_scores=False,
        parallel=False,
    ):
        super(RPEConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            if block == 'self':
                layers.append(RPETransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
            else:
                layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores
        self.parallel = parallel

    def forward(
        self,
        feats0,
        feats1,
        embeddings0,
        embeddings1,
        masks0=None,
        masks1=None,
        conditioner=None,
        points0=None,
        points1=None,
    ):
        attention_scores = []
        diagnostics = {}
        if conditioner is not None:
            if points0 is None or points1 is None:
                raise ValueError('conditioned attention requires both point clouds')
            context0 = conditioner.prepare(points0)
            context1 = conditioner.prepare(points1)
        self_index = 0
        cross_index = 0
        for i, block in enumerate(self.blocks):
            if block == 'self':
                bias0 = None
                bias1 = None
                if conditioner is not None:
                    bias0 = conditioner.self_attention_bias(
                        feats0, context0, self_index
                    )
                    bias1 = conditioner.self_attention_bias(
                        feats1, context1, self_index
                    )
                feats0, scores0 = self.layers[i](
                    feats0,
                    feats0,
                    embeddings0,
                    memory_masks=masks0,
                    attention_bias=bias0,
                )
                feats1, scores1 = self.layers[i](
                    feats1,
                    feats1,
                    embeddings1,
                    memory_masks=masks1,
                    attention_bias=bias1,
                )
                self_index += 1
            else:
                bias01 = None
                bias10 = None
                if conditioner is not None and conditioner.overlap_attention:
                    logits0, logits1 = conditioner.overlap_logits(
                        feats0, feats1, context0, context1
                    )
                    bias01, bias10 = conditioner.cross_attention_bias(
                        logits0, logits1, cross_index
                    )
                    bias01 = bias01.to(dtype=feats0.dtype)
                    bias10 = bias10.to(dtype=feats1.dtype)
                    diagnostics['ref_overlap_logits'] = logits0
                    diagnostics['src_overlap_logits'] = logits1
                if self.parallel:
                    new_feats0, scores0 = self.layers[i](
                        feats0,
                        feats1,
                        memory_masks=masks1,
                        attention_bias=bias01,
                    )
                    new_feats1, scores1 = self.layers[i](
                        feats1,
                        feats0,
                        memory_masks=masks0,
                        attention_bias=bias10,
                    )
                    feats0 = new_feats0
                    feats1 = new_feats1
                else:
                    feats0, scores0 = self.layers[i](
                        feats0,
                        feats1,
                        memory_masks=masks1,
                        attention_bias=bias01,
                    )
                    feats1, scores1 = self.layers[i](
                        feats1,
                        feats0,
                        memory_masks=masks0,
                        attention_bias=bias10,
                    )
                cross_index += 1
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            if conditioner is not None:
                return feats0, feats1, attention_scores, diagnostics
            return feats0, feats1, attention_scores
        else:
            if conditioner is not None:
                return feats0, feats1, diagnostics
            return feats0, feats1


class LRPEConditionalTransformer(nn.Module):
    def __init__(
        self,
        blocks,
        d_model,
        num_heads,
        num_embeddings,
        dropout=None,
        activation_fn='ReLU',
        return_attention_scores=False,
    ):
        super(LRPEConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            if block == 'self':
                layers.append(
                    LRPETransformerLayer(
                        d_model, num_heads, num_embeddings, dropout=dropout, activation_fn=activation_fn
                    )
                )
            else:
                layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores

    def forward(self, feats0, feats1, emb_indices0, emb_indices1, masks0=None, masks1=None):
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers[i](feats0, feats0, emb_indices0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, emb_indices1, memory_masks=masks1)
            else:
                feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1
