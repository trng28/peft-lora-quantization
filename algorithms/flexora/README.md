## Implementation
```
git clone https://github.com/trng28/peft-lora-quantization
cd peft-lora-quantization/algorithms/flexora
```
## Installation
```
pip install -r requirements.txt
```

## Configuration - `config.yaml`
```yaml
model:
  name: "meta-llama/Llama-3.2-1B"
  num_layers: 32

lora:
  r: 8
  lora_alpha: 16
  target_modules: ["q_proj", "v_proj"]
  lora_dropout: 0.05

search:
  search_steps: 100
  lr_model: 1.0e-4
  lr_alpha: 3.0e-2
  batch_size: 2
  max_length: 512

finetune:
  epochs: 10
  lr: 1e-4
  batch_size: 4
```


## Search layer for fine-tuning stage
```
python main.py
```

## Finetune
```
python finetune.py
```
> `latest version: 1.0`
## Reference
```bib
@inproceedings{wei2025flexora,
  title={Flexora: Flexible low-rank adaptation for large language models},
  author={Wei, Chenxing and Shu, Yao and He, Ying Tiffany and Yu, Fei},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={14643--14682},
  year={2025}
}
```