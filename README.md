# MIM

This repo provides a reference implementation of MIM as described in the paper:
Enhancing Information Diffusion Prediction with Cross-Cascade Hypergraph and Past-Future Knowledge Distillation

## Dependencies

Install the dependencies via [Anaconda](https://www.anaconda.com/):

+ Python (>=3.10)
+ PyTorch (>=2.1.1)
+ NumPy (>=1.26.3)
+ Scipy (>=1.12.0)
+ torch-geometric(>=2.4.0)
+ tqdm(>=4.66.1)

```python
# create virtual environment
conda create --name MIMIDP python=3.10.12

# activate environment
conda activate MIMIDP

# install other dependencies
pip install -r requirements.txt
```

## Dataset

See some sample cascades in `./data`.

## Usage

Here we provide the implementation of MIMIDP along with twitter dataset.

+ To train and evaluate on Twitter:
  
  ```python
  python run.py -data_name=twitter
  ```
  
  More running options are described in the codes, e.g., `-data_name= douban`

## Folder Structure

MIMIDP

```
└── data: # The file includes datasets
    ├── twitter
       ├── cascades.txt       # original data
       ├── cascadetrain.txt   # training set
       ├── cascadevalid.txt   # validation set
       ├── cascadetest.txt    # testing data
       ├── edges.txt          # social network
       ├── idx2u.pickle       # idx to user_id
       ├── u2idx.pickle       # user_id to idx
    ├──douban
    ├──memetracker
    ├──weibo

└── model: # The file includes each part of the modules in MIMIDP.
    ├── HGAT.py # The core source code of Convolution.
    ├── model.py # The core source code of MIMIDP.
    ├── TransformerBlock.py # The core source code of time-aware attention.

└── utils: # The file includes each part of basic modules (e.g., metrics, earlystopping).
    ├── EarlyStopping.py  # The core code of the early stopping operation.
    ├── Metrics.py        # The core source code of metrics.
    ├── graphConstruct.py # The core source code of building hypergraph.
    ├── parsers.py        # The core source code of parameter settings.
    ├── util.py           # Transfer variables to the GPU or CPU.
└── Constants.py:    
└── dataLoader.py:     # Data loading.
└── run.py:            # Run the model.
└── Optim.py:          # Optimization.
```
