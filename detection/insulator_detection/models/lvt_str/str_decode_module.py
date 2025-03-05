import torch
from torch import nn as nn, Tensor
from typing import Optional
from pytorch_lightning.utilities import rank_zero_info
from .modules import DecoderLayer, Decoder
from .clip import clip


class STRDecodeModule(nn.Module):
    def __init__(self, dec_num_heads, dec_mlp_ratio, dec_depth,
                 dropout, use_share_dim, max_label_length,
                 use_language_model_fusion, image_detach, charset_adapter,
                 context_length, cross_token_embeding, clip_cls_eot_feature,
                 freeze_language_backbone, tokenizer, clip_model):
        super().__init__()

        self.use_share_dim = use_share_dim
        self.use_language_model_fusion = use_language_model_fusion
        self.image_detach = image_detach
        self.charset_adapter = charset_adapter
        self.context_length = context_length
        self.cross_token_embeding = cross_token_embeding
        self.clip_cls_eot_feature = clip_cls_eot_feature
        self.freeze_language_backbone = freeze_language_backbone

        self.tokenizer = tokenizer
        self.clip_model = clip_model

        vis_embed_dim = self.clip_model.text_projection.shape[-1] if self.use_share_dim else \
            self.clip_model.visual.proj.shape[0]
        rank_zero_info("The dimension of the visual decoder is {}.".format(vis_embed_dim))
        decoder_layer = DecoderLayer(vis_embed_dim, dec_num_heads, vis_embed_dim * dec_mlp_ratio, dropout)
        self.visual_decoder = Decoder(decoder_layer, dec_depth, norm=nn.LayerNorm(vis_embed_dim),
                                      embed_dim=vis_embed_dim,
                                      dropout=dropout,
                                      num_classes=len(self.tokenizer) - 2,
                                      charset_size=len(self.tokenizer),
                                      max_label_length=max_label_length + 1)
        if self.use_language_model_fusion:
            cross_embed_dim = self.clip_model.text_projection.shape[-1]
            decoder_layer = DecoderLayer(cross_embed_dim, dec_num_heads, cross_embed_dim * dec_mlp_ratio, dropout)
            self.fusion_decoder = Decoder(decoder_layer, dec_depth, norm=nn.LayerNorm(cross_embed_dim),
                                         embed_dim=cross_embed_dim,
                                         dropout=dropout,
                                         num_classes=len(self.tokenizer) - 2,
                                         charset_size=len(self.tokenizer),
                                         max_label_length=max_label_length + 1)

    def visual_decode(self, tgt: torch.Tensor, memory: torch.Tensor,
                      tgt_query: Optional[Tensor] = None, tgt_query_mask: Optional[Tensor] = None,
                      content_mask: Optional[Tensor] = None, tgt_padding_mask: Optional[Tensor] = None, ):
        return self.visual_decoder(tgt, memory, tgt_query, tgt_query_mask, content_mask, tgt_padding_mask)

    def fusion_decode(self, prev_logits, tgt: torch.Tensor, memory: torch.Tensor,
                     tgt_query: Optional[Tensor] = None, tgt_query_mask: Optional[Tensor] = None,
                     content_mask: Optional[Tensor] = None, tgt_padding_mask: Optional[Tensor] = None,
                     cross_memory=None):
        if cross_memory is None:
            cross_memory = self.encoder_cross_modal_feature(prev_logits, memory)

        return self.fusion_decoder(tgt, cross_memory, tgt_query, tgt_query_mask, content_mask, tgt_padding_mask)

    def encoder_cross_modal_feature(self, prev_logits, image_feat):
        prev_logits = prev_logits.detach().clone()
        image_features = image_feat.detach().clone() if self.image_detach else image_feat
        if not self.use_share_dim:
            image_features = torch.matmul(image_features, self.clip_model.visual.proj)

        # get previous predictions
        probs = prev_logits.softmax(-1)
        # adapt for the test charset, CLIP is not sensitive to uppercase or symbols
        captions, _ = self.tokenizer.decode_fast(probs, charset_adapter=self.charset_adapter)
        text = clip.tokenize(captions, context_length=self.context_length, truncate=True).to(image_feat.device)

        # return all text features
        if self.freeze_language_backbone:
            with torch.no_grad():
                text_features = self.clip_model.token_embedding(text) if self.cross_token_embeding else \
                    self.clip_model.encode_text(text, eot=self.clip_cls_eot_feature)
        else:
            text_features = self.clip_model.token_embedding(text) if self.cross_token_embeding else \
                self.clip_model.encode_text(text, eot=self.clip_cls_eot_feature)
        if self.clip_cls_eot_feature:
            text_features = torch.unsqueeze(text_features, dim=1)

        # concat image and tex features
        cat_features = torch.cat([image_features, text_features], dim=1)
        return cat_features
