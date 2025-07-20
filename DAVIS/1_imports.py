# !pip install PyTDC rdkit-pypi torch-geometric pandas tqdm
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
import numpy as np
import os
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw
from tdc.multi_pred import DTI
import pandas as pd

