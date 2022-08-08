import yaml
from types import SimpleNamespace
import pandas as pd
import numpy as np
import json

with open('/home/pooja/PycharmProjects/amex_default_kaggle/codes/config.yaml', 'r') as f:
    config = SimpleNamespace(**yaml.safe_load(f))