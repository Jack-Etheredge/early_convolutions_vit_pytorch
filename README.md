# early_convolutions_vit_pytorch
(Unofficial) PyTorch implementation of the paper "Early Convolutions Help Transformers See Better"

Example usage can be found in [this notebook](notebooks/cats_and_dogs_early_conv.ipynb).

This model does appear to outperform the original ViT paper for the same amount of training computation (comparable flops from 1 fewer transformer block and same number of training epochs.)

As a starting point for the original ViT ("An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale") implementation in PyTorch, I used Phil Wang's repo [https://github.com/lucidrains/vit-pytorch/](https://github.com/lucidrains/vit-pytorch/).

Both notebooks have device hard-coded to 'cuda'. I tried training on CPU and got more than a 60x speed up switching to an RTX 2070. If you want to train on CPU for a few epochs, then you'll need to switch that out in the notebook(s).

## Bibtex paper citations:
```bibtex
@misc{xiao2021early,
      title={Early Convolutions Help Transformers See Better}, 
      author={Tete Xiao and Mannat Singh and Eric Mintun and Trevor Darrell and Piotr Dollár and Ross Girshick},
      year={2021},
      eprint={2106.14881},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```bibtex
@misc{dosovitskiy2020image,
    title   = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
    author  = {Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
    year    = {2020},
    eprint  = {2010.11929},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

## Planned updates:
1. Example usage in readme
2. Notebook checks for GPU and uses it if present, otherwise uses CPU
3. Script version of notebook that saves weights and is more flexible regarding input data (intelligently deals with class number, etc)
4. PyTorch Lightning version
5. CLI for model training and weight saving
6. General cleanup and improvements (values from paper are currently hard-coded into the model and there's no testing, logging, etc)