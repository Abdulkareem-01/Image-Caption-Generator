# model_vit47.py

import torch
import torch.nn as nn
from transformers import ViTModel, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


class QFormerCaptionModel(nn.Module):
    def __init__(self, num_query_tokens=16):
        super().__init__()

        # -------- Vision Encoder (Frozen ViT) --------
        self.vit = ViTModel.from_pretrained(
            "google/vit-base-patch16-224"
        )
        for p in self.vit.parameters():
            p.requires_grad = False

        vit_hidden = self.vit.config.hidden_size  # 768

        # -------- Language Decoder (Flan-T5-Base) --------
        self.t5 = T5ForConditionalGeneration.from_pretrained(
            "google/flan-t5-base"
        )

        # Freeze everything
        for p in self.t5.parameters():
            p.requires_grad = False

        # Unfreeze last 2 decoder layers (as used in training)
        for name, param in self.t5.named_parameters():
            if "decoder.block.10" in name or "decoder.block.11" in name:
                param.requires_grad = True

        t5_hidden = self.t5.config.d_model  # 768

        # -------- Q-Former --------
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_query_tokens, vit_hidden)
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=vit_hidden,
            num_heads=8,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(vit_hidden, vit_hidden * 4),
            nn.GELU(),
            nn.Linear(vit_hidden * 4, vit_hidden)
        )

        self.proj = nn.Linear(vit_hidden, t5_hidden)

    # ---------------- TRAINING ----------------
    def forward(self, pixel_values, labels=None):
        B = pixel_values.size(0)

        with torch.no_grad():
            vit_feats = self.vit(
                pixel_values=pixel_values
            ).last_hidden_state

        queries = self.query_tokens.expand(B, -1, -1)
        q_out, _ = self.cross_attn(queries, vit_feats, vit_feats)
        q_out = q_out + self.ffn(q_out)

        encoder_hidden_states = self.proj(q_out)

        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_hidden_states
        )

        return self.t5(
            encoder_outputs=encoder_outputs,
            labels=labels
        )

    # ---------------- INFERENCE ----------------
    @torch.no_grad()
    def generate(self, pixel_values, tokenizer, max_length=40):
        self.eval()
        B = pixel_values.size(0)

        vit_feats = self.vit(
            pixel_values=pixel_values
        ).last_hidden_state

        queries = self.query_tokens.expand(B, -1, -1)
        q_out, _ = self.cross_attn(queries, vit_feats, vit_feats)
        q_out = q_out + self.ffn(q_out)

        encoder_hidden_states = self.proj(q_out)

        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_hidden_states
        )

        decoder_input_ids = tokenizer(
            ["describe what is happening in the image, focusing on visible actions and interactions if present:"] * B,
            return_tensors="pt",
            padding=True
        ).input_ids.to(pixel_values.device)

        generated = self.t5.generate(
            input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            max_length=max_length,
            num_beams=5,
            do_sample=False,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

        return generated
