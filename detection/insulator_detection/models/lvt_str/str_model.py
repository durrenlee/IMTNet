# coding=utf-8
# modified based on https://github.com/VamosC/CLIP4STR/

import os
import math
import sys

import numpy as np
from itertools import permutations
from typing import Sequence, Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.types import STEP_OUTPUT

o_path = os.getcwd()
sys.path.append(o_path)
from .clip import clip

from .base import CrossEntropySystem
from .modules import  modify_attn_mask
from .str_decode_module import STRDecodeModule
from mmdet.models.backbones.multiscalelgtformer import MultiScaleLgtFormer

# an alternative choice when the input argument is not valid
# CLIP_PATH = '/PUT/YOUR/PATH/HERE/pretrained/clip''
CLIP_PATH = '/root/pretrained'
if not os.path.exists(CLIP_PATH):
    CLIP_PATH = '/root/pretrained'
assert os.path.exists(CLIP_PATH)


class InsulatorSTRModel(CrossEntropySystem):

    def __init__(self, charset_train: str, charset_test: str, max_label_length: int,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float,
                 img_size: Sequence[int], patch_size: Sequence[int], embed_dim: int,
                 enc_num_heads: int, enc_mlp_ratio: int, enc_depth: int,
                 dec_num_heads: int, dec_mlp_ratio: int, dec_depth: int,
                 perm_num: int, perm_forward: bool, perm_mirrored: bool,
                 decode_ar: bool, refine_iters: int, dropout: float, **kwargs: Any) -> None:
        super().__init__(charset_train, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.save_hyperparameters()

        self.max_label_length = max_label_length
        self.decode_ar = decode_ar
        self.refine_iters = refine_iters
        self.coef_lr = kwargs["coef_lr"] if "coef_lr" in kwargs.keys() else 1.0
        self.coef_wd = kwargs["coef_wd"] if "coef_wd" in kwargs.keys() else 1.0
        self.image_freeze_nlayer = kwargs["image_freeze_nlayer"] if "image_freeze_nlayer" in kwargs.keys() else -1
        self.text_freeze_nlayer = kwargs["text_freeze_nlayer"] if "text_freeze_nlayer" in kwargs.keys() else -1
        self.freeze_language_backbone = self.text_freeze_nlayer >= 12
        self.freeze_image_backbone = self.image_freeze_nlayer >= 12
        self.use_language_model_fusion = kwargs[
            "use_language_model_fusion"] if "use_language_model_fusion" in kwargs.keys() else False
        self.context_length = kwargs["context_length"] if "context_length" in kwargs.keys() else 20
        self.cross_loss_w = kwargs["cross_loss_w"] if "cross_loss_w" in kwargs.keys() else 1.0
        self.use_share_dim = kwargs["use_share_dim"] if "use_share_dim" in kwargs.keys() else True
        self.cross_gt_context = kwargs["cross_gt_context"] if "cross_gt_context" in kwargs.keys() else True
        self.cross_cloze_mask = kwargs["cross_cloze_mask"] if "cross_cloze_mask" in kwargs.keys() else False
        self.cross_correct_once = kwargs["cross_correct_once"] if "cross_correct_once" in kwargs.keys() else False
        self.image_detach = kwargs["image_detach"] if "image_detach" in kwargs.keys() else True
        self.cross_token_embeding = kwargs["cross_token_embeding"] if "cross_token_embeding" in kwargs.keys() else False
        self.clip_cls_eot_feature = kwargs["clip_cls_eot_feature"] if "clip_cls_eot_feature" in kwargs.keys() else False

        rank_zero_info("\n config of VL4STR: \n"
                       "\t image_freeze_nlayer: {}, text_freeze_nlayer: {}, freeze_language_backbone: {}, freeze_image_backbone: {} \n"
                       "\t use_language_model_fusion: {}, context_length: {}, cross_token_embeding: {}, cross_loss_weight: {} \n"
                       "\t use_share_dim: {}, image_detach: {}, clip_cls_eot_feature: {} \n"
                       "\t cross_gt_context: {}, cross_cloze_mask: {}, \n".format(
            self.image_freeze_nlayer, self.text_freeze_nlayer, self.freeze_language_backbone,
            self.freeze_image_backbone,
            self.use_language_model_fusion, self.context_length, self.cross_token_embeding, self.cross_loss_w,
            self.use_share_dim, self.image_detach, self.clip_cls_eot_feature,
            self.cross_gt_context, self.cross_cloze_mask)
        )

        # image encoder
        self.image_encoder = MultiScaleLgtFormer(
            img_size=kwargs['lgtvit_img_size'],
            patch_size=kwargs['lgtvit_patch_size'],
            in_chans=kwargs['lgtvit_in_chans'],
            num_classes=kwargs['lgtvit_num_classes'],
            embed_dim=kwargs['lgtvit_embed_dim'],
            depths=kwargs['lgtvit_depths'],
            groups=kwargs['lgtvit_groups'],
            num_heads=kwargs['lgtvit_num_heads'],
            kernel_size=kwargs['lgtvit_kernel_size'],
            dilation=kwargs['lgtvit_dilation'],
            mlp_ratio=kwargs['lgtvit_mlp_ratio'],
            qkv_bias=kwargs['lgtvit_qkv_bias'],
            qk_scale=kwargs['lgtvit_qk_scale'],
            drop=kwargs['lgtvit_drop'],
            attn_drop=kwargs['lgtvit_attn_drop'],
            drop_path=kwargs['lgtvit_drop_path'],
            norm_layer=kwargs['lgtvit_norm_layer'],
            merging_way=kwargs['lgtvit_merging_way'],
            patch_way=kwargs['lgtvit_patch_way'],
            dilate_attention=kwargs['lgtvit_dilate_attention'],
            downsamples=kwargs['lgtvit_downsamples'],
            cpe_per_satge=kwargs['lgtvit_cpe_per_satge'],
            cpe_per_block=kwargs['lgtvit_cpe_per_block'],
            offset_scale=kwargs['lgtvit_offset_scale'],
            dw_kernel_size=kwargs['lgtvit_dw_kernel_size'],
            center_feature_scale=kwargs['lgtvit_center_feature_scale'],
            remove_center=kwargs['lgtvit_remove_center'],
            output_bias=kwargs['lgtvit_output_bias'],
            without_pointwise=kwargs['lgtvit_without_pointwise'],
            out_indices=kwargs['lgtvit_out_indices'],
            task=kwargs['lgtvit_task'],
            init_cfg=kwargs['lgtvit_init_cfg'],
        )

        assert "clip_pretrained" in kwargs.keys()
        if not os.path.exists(kwargs["clip_pretrained"]):
            kwargs["clip_pretrained"] = os.path.join(CLIP_PATH, os.path.basename(kwargs["clip_pretrained"]))
            print(">>> Try to load CLIP model from {}".format(kwargs["clip_pretrained"]))
            assert os.path.exists(kwargs["clip_pretrained"])
        # load CLIP model
        clip_model, _ = clip.load(name=kwargs["clip_pretrained"], device='cpu')
        self.clip_model = clip_model.float()

        # modify the attention mask according to context length
        self.clip_model.transformer.apply(lambda m: modify_attn_mask(m, context_length=self.context_length))
        self.freeze_cip_layers(self.image_freeze_nlayer, self.text_freeze_nlayer)

        # Perm/attn mask stuff
        self.rng = np.random.default_rng()
        self.max_gen_perms = perm_num // 2 if perm_mirrored else perm_num
        self.perm_forward = perm_forward
        self.perm_mirrored = perm_mirrored

        # fusion decoder module
        self.fusion_decoder = STRDecodeModule(
            dec_num_heads, dec_mlp_ratio, dec_depth,
            dropout, self.use_share_dim, max_label_length,
            self.use_language_model_fusion, self.image_detach, self.charset_adapter,
            self.context_length, self.cross_token_embeding, self.clip_cls_eot_feature,
            self.freeze_language_backbone, self.tokenizer, self.clip_model
        )

    def encode(self, img: torch.Tensor):
        memory = self.image_encoder(img)
        return memory if not self.clip_cls_eot_feature else torch.unsqueeze(memory, dim=1)

    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        testing = max_length is None
        max_length = self.max_label_length if max_length is None else min(max_length, self.max_label_length)
        bs = images.shape[0]
        # +1 for <eos> at end of sequence.
        num_steps = max_length + 1
        memory = self.encode(images)

        # Query positions up to `num_steps`
        vis_pos_queries = self.fusion_decoder.visual_decoder.pos_queries[:, :num_steps].expand(bs, -1, -1)
        crs_pos_queries = self.fusion_decoder.fusion_decoder.pos_queries[:, :num_steps].expand(bs, -1,
                                                                                               -1) if self.use_language_model_fusion else None

        # a left-to-right auto-regressive mask, special case for the forward permutation
        content_mask = query_mask = torch.triu(torch.full((num_steps, num_steps), float('-inf'), device=self._device),
                                               1)
        bos = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self._device)

        ### First Decoding Stage - Coarse-grained Predicted Text ###
        # autoregressive decoding: preventing tokens from attending to future tokens
        if self.decode_ar:
            # tgt_in:
            # The tensor starts with all values set to pad_id,
            # meaning it is an empty target sequence (to be filled later).
            # It stores the growing predicted sequence.
            tgt_in = torch.full((bs, num_steps), self.pad_id, dtype=torch.long, device=self._device)
            #  starts with a beginning-of-sequence (<bos>) token.
            tgt_in[:, 0] = self.bos_id

            logits = []
            all_visual_vec = []
            # Step-by-step decoding:
            # A loop iterates over num_steps, feeding previously predicted tokens into the decoder.
            for i in range(num_steps):
                j = i + 1  # next token index
                '''
                Efficient decoding( A single query at position i is used):
                    input the context up to the ith token. We use only one query (at position = i) at a 
                    time. This works because of the lookahead masking effect of the canonical (forward) AR context.
                    Past tokens have no access to future tokens, hence are fixed once computed.
                '''
                p_i, visual_vec = self.fusion_decoder.visual_decode(tgt_in[:, :j], memory,
                                                                    tgt_query=vis_pos_queries[:, i:j],
                                                                    tgt_query_mask=query_mask[i:j, :j],
                                                                    content_mask=content_mask[:j, :j], )

                # the next token probability is in the output's ith token position
                logits.append(p_i)
                all_visual_vec.append(visual_vec.clone())

                # The next token is greedily selected (argmax) and added to target input: tgt_in.
                if j < num_steps:
                    # greedy decode. add the next token index to the target input
                    tgt_in[:, j] = p_i.squeeze().argmax(-1)
                    # If <eos> (end-of-sequence) is reached for all batch elements, decoding stops early.
                    if testing and (tgt_in == self.eos_id).any(dim=-1).all():
                        break

            # The accumulated logits and visual feature vectors are stored.
            logits = torch.cat(logits, dim=1)
            visual_vec = torch.cat(all_visual_vec, dim=1)

        else:
            # No prior context, so input is just <bos>. We query all positions.
            # tgt_in = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self._device)
            logits, visual_vec = self.fusion_decoder.visual_decode(bos, memory, tgt_query=vis_pos_queries)

        ### Second Decoding Stage - Fine-grained Predicted Text ###
        if self.use_language_model_fusion:
            crs_num_steps = logits.shape[1]

            # encode coarse-grained predicted text and
            # fuze visual feature and coarse-grained text feature
            cross_modal_memory = self.fusion_decoder.encoder_cross_modal_feature(logits, memory)

            # Step-by-step autoregressive decoding:
            cross_logits = []
            all_cross_vec = []
            for i in range(crs_num_steps):
                j = i + 1  # next token index
                p_i, cross_vec = self.fusion_decoder.fusion_decode(logits, tgt_in[:, :j], memory,
                                                                   tgt_query=crs_pos_queries[:, i:j],
                                                                   tgt_query_mask=query_mask[i:j, :j],
                                                                   content_mask=content_mask[:j, :j],
                                                                   cross_memory=cross_modal_memory)
                cross_logits.append(p_i)
                all_cross_vec.append(cross_vec.clone())
                if j < crs_num_steps:
                    tgt_in[:, j] = p_i.squeeze().argmax(-1)
            cross_logits = torch.cat(cross_logits, dim=1)
            cross_vec = torch.cat(all_cross_vec, dim=1)

        ### Iterative Refinement ###
        if self.refine_iters:
            '''
            A cloze mask is derived from the AR forward mask by selectively unmasking certain future tokens, 
            allowing bidirectional context within the predicted sequence to improve accuracy by refining 
            earlier predictions while still respecting <eos> constraints.
            '''
            query_mask[torch.triu(torch.ones(num_steps, num_steps, dtype=torch.bool, device=self._device), 2)] = 0
            bos = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self._device)
            for i in range(self.refine_iters):
                # Prior context is the previous output.
                tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                tgt_padding_mask = ((tgt_in == self.eos_id).cumsum(-1) > 0)  # mask tokens beyond the first EOS token.
                logits, visual_vec = self.fusion_decoder.visual_decode(tgt_in, memory,
                                                                       tgt_query=vis_pos_queries,
                                                                       tgt_query_mask=query_mask[:, :tgt_in.shape[1]],
                                                                       content_mask=content_mask,
                                                                       tgt_padding_mask=tgt_padding_mask, )
                if self.use_language_model_fusion:
                    tgt_in = torch.cat([bos, cross_logits[:, :-1].argmax(-1)], dim=1)
                    tgt_padding_mask = ((tgt_in == self.eos_id).cumsum(-1) > 0)
                    cross_logits, cross_vec = self.fusion_decoder.fusion_decode(logits, tgt_in, memory,
                                                                                tgt_query=crs_pos_queries,
                                                                                tgt_query_mask=query_mask[:,
                                                                                               :tgt_in.shape[1]],
                                                                                content_mask=content_mask,
                                                                                tgt_padding_mask=tgt_padding_mask, )

        logits = cross_logits
        return logits

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        # converts text labels into token sequences
        tgt = self.tokenizer.encode(labels, self._device)

        # extracts visual features
        image_memory = self.encode(images)

        # prepare target sequences
        # with PSM to generate different permutations of the target sequence for training with reordering.
        tgt_perms = self.gen_tgt_perms(tgt)
        # the input sequence to the decoder (without [EOS]).
        tgt_in = tgt[:, :-1]  # remove [EOS] token
        # the expected output sequence (without [BOS]).
        tgt_out = tgt[:, 1:]  # remove [BOS] token

        #  compute attention masks
        # identifies padding and [EOS] tokens to avoid computing loss on them.
        tgt_padding_mask = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)
        bs = images.shape[0]
        bos = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self._device)

        # defines a cloze mask for fine-grained prediction that
        # selectively allows certain future tokens to be attended.
        L = tgt_in.shape[1]
        cloze_content_mask = cloze_query_mask = torch.triu(torch.full((L, L), float('-inf'), device=self._device), 1)
        cloze_query_mask[torch.triu(torch.ones(L, L, dtype=torch.bool, device=self._device), 2)] = 0

        # init loss vars
        # for accumulating the total loss
        loss = 0
        # for tracking the total number of valid tokens
        loss_numel = 0

        #  n: the number of non-padding tokens in tgt_out:
        #  actual token seq length for variable labels length
        n = (tgt_out != self.pad_id).sum().item()

        # compute Loss for permuted target sequences(used for training robust sequence prediction)
        for i, perm in enumerate(tgt_perms):
            # generates attention masks based on the current permutation:perm
            content_mask, query_mask = self.generate_attn_masks(perm)
            # decoding based on token seqs and image features for coarse-grained text prediction
            visual_logits, visual_vec = self.fusion_decoder.visual_decode(tgt_in, image_memory, tgt_query_mask=query_mask,
                                                                          content_mask=content_mask,
                                                                          tgt_padding_mask=tgt_padding_mask, )

            # computes cross-entropy loss between predicted tokens (visual_logits) and ground-truth tokens (tgt_out)
            # Loss is weighted by n to account for variable-length sequences
            loss += n * F.cross_entropy(visual_logits.flatten(end_dim=1), tgt_out.flatten(), ignore_index=self.pad_id)

            if self.use_language_model_fusion:
                # self.cross_gt_context:True, still use original tgt_in
                cross_tgt_in = tgt_in if self.cross_gt_context else torch.cat([bos, visual_logits[:, :-1].argmax(-1)],
                                                                              dim=1)
                # self.cross_cloze_mask:False, still use query_mask, content_mask from generated permutation
                cross_query_mask = cloze_query_mask if self.cross_cloze_mask else query_mask
                cross_content_mask = cloze_content_mask if self.cross_cloze_mask else content_mask
                # decoding based on visual logits, token seqs
                cross_logits, cross_vec = self.fusion_decoder.fusion_decode(visual_logits, cross_tgt_in, image_memory,
                                                                            tgt_query_mask=cross_query_mask,
                                                                            content_mask=cross_content_mask,
                                                                            tgt_padding_mask=tgt_padding_mask, )
                # minimize (visual loss + cross loss)
                loss += self.cross_loss_w * n * F.cross_entropy(cross_logits.flatten(end_dim=1), tgt_out.flatten(),
                                                                ignore_index=self.pad_id)

            loss_numel += n
            # After the second iteration (i.e. done with canonical and reverse orderings),
            # remove the [EOS] tokens for the succeeding perms
            if i == 1:
                tgt_out = torch.where(tgt_out == self.eos_id, self.pad_id, tgt_out)
                n = (tgt_out != self.pad_id).sum().item()

        loss /= loss_numel

        # self.log('loss', loss)
        return loss

    def gen_tgt_perms(self, tgt):
        """Generate shared permutations for the whole batch.
           This works because the same attention mask can be used for the shorter sequences
           because of the padding mask.
           An example fo perms with string length 8
           >>> tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],   # canonical order
                        [0, 9, 8, 7, 6, 5, 4, 3, 2, 1],  # reverse order
                        [0, 7, 5, 4, 8, 3, 2, 6, 1, 9],
                        [0, 1, 6, 2, 3, 8, 4, 5, 7, 9],
                        [0, 7, 8, 6, 3, 1, 5, 2, 4, 9],
                        [0, 4, 2, 5, 1, 3, 6, 8, 7, 9]], device='cuda:0')
        """
        # We don't permute the position of BOS, we permute EOS separately
        max_num_chars = tgt.shape[1] - 2
        # Special handling for 1-character sequences
        if max_num_chars == 1:
            return torch.arange(3, device=self._device).unsqueeze(0)
        perms = [torch.arange(max_num_chars, device=self._device)] if self.perm_forward else []
        # Additional permutations if needed
        max_perms = math.factorial(max_num_chars)
        if self.perm_mirrored:
            max_perms //= 2
        num_gen_perms = min(self.max_gen_perms, max_perms)
        # For 4-char sequences and shorter, we generate all permutations and sample from the pool to avoid collisions
        # Note that this code path might NEVER get executed since the labels in a mini-batch typically exceed 4 chars.
        if max_num_chars < 5:
            # Pool of permutations to sample from. We only need the first half (if complementary option is selected)
            # Special handling for max_num_chars == 4 which correctly divides the pool into the flipped halves
            if max_num_chars == 4 and self.perm_mirrored:
                selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]
            else:
                selector = list(range(max_perms))
            perm_pool = torch.as_tensor(list(permutations(range(max_num_chars), max_num_chars)), device=self._device)[
                selector]
            # If the forward permutation is always selected, no need to add it to the pool for sampling
            if self.perm_forward:
                perm_pool = perm_pool[1:]
            perms = torch.stack(perms)
            if len(perm_pool):
                i = self.rng.choice(len(perm_pool), size=num_gen_perms - len(perms), replace=False)
                perms = torch.cat([perms, perm_pool[i]])
        else:
            perms.extend(
                [torch.randperm(max_num_chars, device=self._device) for _ in range(num_gen_perms - len(perms))])
            perms = torch.stack(perms)
        if self.perm_mirrored:
            # Add complementary pairs
            comp = perms.flip(-1)
            # Stack in such a way that the pairs are next to each other.
            perms = torch.stack([perms, comp]).transpose(0, 1).reshape(-1, max_num_chars)
        # NOTE:
        # The only meaningful way of permuting the EOS position is by moving it one character position at a time.
        # However, since the number of permutations = T! and number of EOS positions = T + 1, the number of possible EOS
        # positions will always be much less than the number of permutations (unless a low perm_num is set).
        # Thus, it would be simpler to just train EOS using the full and null contexts rather than trying to evenly
        # distribute it across the chosen number of permutations.
        # Add position indices of BOS and EOS
        bos_idx = perms.new_zeros((len(perms), 1))
        eos_idx = perms.new_full((len(perms), 1), max_num_chars + 1)
        perms = torch.cat([bos_idx, perms + 1, eos_idx], dim=1)
        # Special handling for the reverse direction. This does two things:
        # 1. Reverse context for the characters
        # 2. Null context for [EOS] (required for learning to predict [EOS] in NAR mode)
        if len(perms) > 1:
            perms[1, 1:] = max_num_chars + 1 - torch.arange(max_num_chars + 1, device=self._device)

        # print(f'perms:{perms}')
        return perms

    def generate_attn_masks(self, perm):
        """Generate attention masks given a sequence permutation (includes pos. for bos and eos tokens)
        :param perm: the permutation sequence. i = 0 is always the BOS
        :return: lookahead attention masks
        """
        sz = perm.shape[0]
        mask = torch.zeros((sz, sz), device=self._device)
        for i in range(sz):
            query_idx = perm[i]
            masked_keys = perm[i + 1:]
            mask[query_idx, masked_keys] = float('-inf')
        content_mask = mask[:-1, :-1].clone()
        mask[torch.eye(sz, dtype=torch.bool, device=self._device)] = float('-inf')  # mask "self"
        query_mask = mask[1:, :-1]
        return content_mask, query_mask

    def _freeze_backbones(self):
        """set frozen backbones to eval mode"""
        for name, mod in self.clip_model.named_modules():
            if name.startswith("visual.transformer.resblocks."):
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num < self.image_freeze_nlayer:
                    mod.eval()

                # if self.image_freeze_layer_divisor > 0 and (layer_num + 1) % self.image_freeze_layer_divisor == 0:
                #     mod.eval()
            elif name.startswith("transformer.resblocks."):
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num < self.text_freeze_nlayer:
                    mod.eval()

        if self.freeze_language_backbone:
            self.clip_model.transformer.eval()
            self.clip_model.ln_final.eval()

    def freeze_cip_layers(self, image_freeze_nlayer, text_freeze_nlayer, image_freeze_layer_divisor=-1,
                          image_only_fc=False):
        """
        freeze the parameters of layers with No.layer < image_freeze_nlayer or text_freeze_nlayer,
        """
        assert image_freeze_nlayer <= 12 and text_freeze_nlayer <= 12 and image_freeze_layer_divisor <= 12
        if hasattr(self, "clip_model"):
            if image_freeze_nlayer > -1:
                for name, param in self.clip_model.visual.named_parameters():
                    # top layers always need to train
                    if name.startswith("ln_post.") or name.startswith("proj") or name.startswith(
                            "conv1") or name.startswith("ln_pre"):
                        continue
                    elif name.startswith("transformer.resblocks."):
                        layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                        if layer_num >= image_freeze_nlayer:
                            continue
                    param.requires_grad = False

            if text_freeze_nlayer > -1:
                for name, param in self.clip_model.named_parameters():
                    # top layers always need to train
                    if name.startswith("ln_final.") or name.startswith("text_projection") or name.startswith("visual"):
                        continue
                    elif name.startswith("transformer.resblocks."):
                        layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                        if layer_num >= text_freeze_nlayer:
                            continue
                    param.requires_grad = False

            # freeze the whole backbones and related parameters
            if text_freeze_nlayer >= 12:
                for n, p in self.clip_model.named_parameters():
                    # exclude visual parameters
                    if "visual" not in n:
                        if "transformer" in n or "token_embedding" in n or "ln_final" in n or "text_projection" in n:
                            p.requires_grad = False

            if image_freeze_nlayer >= 12:
                for n, p in self.clip_model.visual.named_parameters():
                    p.requires_grad = False

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if mode:
            self._freeze_backbones()

        return self
