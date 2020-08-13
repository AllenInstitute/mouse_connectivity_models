"""trVAE - Transfer Variational Autoencoders pytorch"""

from .model import *
from .data_loader import CustomDatasetFromAdata
from .pl import *




__author__ = ', '.join([
    'Mohammad Lotfollahi'
])

__email__ = ', '.join([
    'Mohammad.lotfollahi@helmholtz-muenchen.de',
])

from get_version import get_version
__version__ = get_version(__file__)

del get_version
