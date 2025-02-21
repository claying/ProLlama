import torch
import hydra
import transformers
import pytorch_lightning as pl


def get_llama_model(model_name, alphabet, **kwargs):
    model_sizes = {
        'llama-xs': {
            'hidden_size': 512,
            'num_hidden_layers': 6,
            'num_attention_heads': 8,
            'intermediate_size': 4 * 512,
            'n_embd': 512,
            'n_head': 8,
            'n_layer': 6,
        },
        'llama-s': {
            'hidden_size': 768,
            'num_hidden_layers': 12,
            'num_attention_heads': 12,
            'intermediate_size': 4 * 768,
            'n_embd': 768,
            'n_head': 12,
            'n_layer': 12,
        },
    }
    model_size_config = model_sizes[model_name]
    model_config = {
        "rms_norm_eps": 1e-05,
        "max_position_embeddings": 4096,
        "attn_implementation": "sdpa"
    }
    vocab_config = {
        'vocab_size': len(alphabet),
        'bos_token_id': alphabet.cls_idx,
        'eos_token_id': alphabet.eos_idx,
        'pad_token_id': alphabet.padding_idx,
    }

    config = transformers.LlamaConfig(
        **vocab_config, **model_config, **model_size_config, **kwargs
    )
    return transformers.LlamaForCausalLM(config)


class ProLlama(pl.LightningModule):
    def __init__(self, cfg, new_cfg=None):
        super().__init__()

        self.cfg = cfg
        if new_cfg is not None:
            self.cfg = new_cfg

        self.instantiate_datamodule()
        self.instantiate_model(cfg)
        self.save_hyperparameters()

    def instantiate_datamodule(self):
        self._datamodule = hydra.utils.instantiate(self.cfg.datamodule)
        # self._datamodule.setup()
        self.alphabet = self._datamodule.alphabet

    def instantiate_model(self, cfg):
        self.model = hydra.utils.call(cfg.model, alphabet=self.alphabet)

    def forward(self, input_ids):
        return self.model(input_ids=input_ids).logits

    def shared_step(self, batch, batch_idx, phase='train'):
        out = self.model(input_ids=batch, labels=batch)
        loss = out.loss
        on_step = phase == 'train'
        self.log(f"{phase}/loss", loss, on_step=on_step, on_epoch=not on_step, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, phase='train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, phase='val')
        return loss

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.cfg.train.optimizer, self.model.parameters())
        lr_scheduler = hydra.utils.call(self.cfg.train.lr_scheduler, optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }

    @torch.inference_mode()
    def generate(self, num_samples=None, input_ids=None):
        num_samples = self.cfg.sampling.num_samples if num_samples is None else num_samples
        if input_ids is not None and isinstance(input_ids, torch.Tensor):
            num_samples = input_ids.shape[0]
        seq_list = []
        for i in range(0, num_samples, self.cfg.sampling.batch_size):
            batch_size = min(self.cfg.sampling.batch_size, num_samples - i)
            if input_ids is not None and isinstance(input_ids, torch.Tensor):
                input_ids_cur = input_ids[i:i + batch_size]
                input_ids_cur.to(self.device)
            else:
                input_ids_cur = torch.full(
                    (batch_size, 1), self.alphabet.cls_idx, dtype=torch.long, device=self.device
                )
            seq = self._generate(input_ids_cur)
            seq_list.extend(seq)
        return seq_list

    def _generate(self, input_ids):
        batch_size = input_ids.shape[0]
        toks = self.model.generate(
            input_ids,
            do_sample=True,
            top_k=self.cfg.sampling.top_k,
            temperature=self.cfg.sampling.temperature,
            max_length=self.cfg.sampling.max_length,
        )
        seq_list = []
        for i in range(batch_size):
            tok_seq = toks[i][1:-1]
            seq = "".join([self.alphabet.get_tok(tok.item()) for tok in tok_seq])
            seq_list.append(seq)
        return seq_list
