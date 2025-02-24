# ProLlama
A minimal implementation for training a Llama model on protein sequences from scratch with less than 500 lines of code.

The repository largely relies on [Lightning][1], [Hydra][2] and [HuggingFace's Llama][3] implementation.
You can find my tutorial [here][4].

## Install

```bash
pip install torch torchvision lightning hydra-core transformers fair-esm
```

If you use `wandb` to log the metrics, you can optionally install it `pip install wandb`.

## Quick start

To train the model:

```bash
python train.py
```

To generate protein sequences:

```bash
python generate.py model.pretrained_path=${path_to_your_pretrained_model} sampling.num_samples=10
```

[1]: https://lightning.ai/
[2]: https://hydra.cc/
[3]: https://huggingface.co/docs/transformers/main/en/model_doc/llama
[4]: https://dexiong.me/blogs/2025-02-24-lightning-and-hydra/

