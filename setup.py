from setuptools import setup, find_packages

setup(
  name = 'early-convolutions-vit-pytorch',
  packages = find_packages(exclude=['notebooks']),
  version = '0.0.1',
  license='MIT',
  description = 'Early Convolutions Vision Transformer (ViTC) - Pytorch',
  author = 'Jack Etheredge',
  author_email = 'jack.etheredge2@gmail.com',
  url = 'https://github.com/Jack-Etheredge/early_convolutions_vit_pytorch',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'image recognition'
  ],
  install_requires=[
    'einops>=0.3',
    'torch>=1.6',
    'torchvision'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
