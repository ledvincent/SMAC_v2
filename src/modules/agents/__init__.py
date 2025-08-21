REGISTRY = {}

from .hpn_rnn_agent import HPN_RNNAgent
from .hpns_rnn_agent import HPNS_RNNAgent
from .rnn_agent import RNNAgent
from .updet_agent import UPDeT
from .ss_rnn_agent import SS_RNNAgent   
from .att_agent import SS_HF_RNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["hpn_rnn"] = HPN_RNNAgent
REGISTRY["hpns_rnn"] = HPNS_RNNAgent
REGISTRY["updet_agent"] = UPDeT
REGISTRY["ss_rnn"] = SS_RNNAgent
REGISTRY["att_rnn"] = SS_HF_RNNAgent
