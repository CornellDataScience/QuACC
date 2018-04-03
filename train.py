"""
Training script.
"""

from hyperparams import Hyperparams as Hp
from loader import Loader
from model import Model
from tqdm import tqdm

data = Loader(Hp.batch_size)
