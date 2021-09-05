# early_convolutions_vit_pytorch
(Unofficial) PyTorch implementation of the paper "Early Convolutions Help Transformers See Better"

Example usage can be found in [this notebook](notebooks/cats_and_dogs_early_conv.ipynb).

This model does appear to outperform the original ViT paper for the same amount of training computation (comparable flops from 1 fewer transformer block and same number of training epochs.)

As a starting point for the original ViT ("An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale") implementation in PyTorch, I used Phil Wang's repo [https://github.com/lucidrains/vit-pytorch/](https://github.com/lucidrains/vit-pytorch/).

Both notebooks will use the GPU if it's available according to torch. The training is quite slow on CPU. I tried training on CPU and got more than a 60x speed up switching to an RTX 2070 (your speedup will, of course, depend on the CPU and GPU).

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
2. Script version of notebook that saves weights and is more flexible regarding input data (intelligently deals with class number, etc)
3. PyTorch Lightning version
4. CLI for model training and weight saving
5. General cleanup and improvements (values from paper are currently hard-coded into the model and there's no testing, logging, etc)