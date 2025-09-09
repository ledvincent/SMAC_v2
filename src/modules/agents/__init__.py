REGISTRY = {}

from .hpn_rnn_agent import HPN_RNNAgent
from .hpns_rnn_agent import HPNS_RNNAgent
from .rnn_agent import RNNAgent
from .updet_agent import UPDeT
from .ss_rnn_agent import SS_RNNAgent   
from .att_agent import CustomAtt_RNNAgent
from .bias_agent import MHA_QAgent
from .tactical_agent import CAPERNN_ContRoles_HPN
from .role_agent import RoleEmergentAgent
from .role_agent import RoleEmergentAgent as RoleEmergentAgent2

REGISTRY["rnn"] = RNNAgent
REGISTRY["hpn_rnn"] = HPN_RNNAgent
REGISTRY["hpns_rnn"] = HPNS_RNNAgent
REGISTRY["updet_agent"] = UPDeT
REGISTRY["ss_rnn"] = SS_RNNAgent
REGISTRY["att_rnn"] = CustomAtt_RNNAgent
REGISTRY["bias_agent"] = MHA_QAgent
REGISTRY["tactical_rnn"] = CAPERNN_ContRoles_HPN
REGISTRY["role_rnn"] = RoleEmergentAgent
REGISTRY["role_sep_rnn"] = RoleEmergentAgent2