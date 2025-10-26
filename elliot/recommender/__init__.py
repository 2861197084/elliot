"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from .base_recommender_model import BaseRecommenderModel

# Import light-weight, non-TensorFlow recommenders first
from .knn import ItemKNN, UserKNN, AttributeItemKNN, AttributeUserKNN
from .content_based import VSM
from .algebric import SlopeOne
from .unpersonalized import Random, MostPop
from .generic import ProxyRecommender

# Import RP3beta directly to avoid pulling TF-based graph modules
from .graph_based.RP3beta import RP3beta

# Best-effort imports for TF-based or heavy modules. If missing TF or plugins, skip gracefully.
try:
    from .latent_factor_models import BPRMF, BPRMF_batch, WRMF, PureSVD, MF, FunkSVD, PMF, LMF, NonNegMF, FM, LogisticMF, FFM, BPRSlim, Slim, CML, FISM, SVDpp, MF2020, iALS
except Exception:  # pragma: no cover
    pass

try:
    from .neural import DeepFM, DMF, NeuMF, NFM, GMF, NAIS, UserAutoRec, ItemAutoRec, ConvNeuMF, WideAndDeep, ConvMF, NPR
except Exception:  # pragma: no cover
    pass

try:
    from .autoencoders import MultiDAE, MultiVAE, EASER
except Exception:  # pragma: no cover
    pass

try:
    from .visual_recommenders import VBPR, DeepStyle, ACF, DVBPR, VNPR
except Exception:  # pragma: no cover
    pass

try:
    from .knowledge_aware import KaHFM, KaHFMBatch, KaHFMEmbeddings
except Exception:  # pragma: no cover
    pass

try:
    from .graph_based import NGCF, LightGCN
except Exception:  # pragma: no cover
    pass

try:
    from .adversarial import AMF, AMR
except Exception:  # pragma: no cover
    pass

try:
    from .gan import IRGAN, CFGAN
except Exception:  # pragma: no cover
    pass
