from typing import Union, Type, Dict
from .hpn_controller import HPNMAC
from .basic_controller import BasicMAC
from .n_controller import NMAC
from .updet_controller import UPDETController
from .ss_controller import SSMAC
from .custom_controller import CustomMAC

REGISTRY:Dict[str,
              Union[Type[BasicMAC],
                    Type[HPNMAC],
                    Type[NMAC],
                    Type[UPDETController],
                    Type[SSMAC],
                    Type[CustomMAC]]
    ] = {}

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["n_mac"] = NMAC
REGISTRY["hpn_mac"] = HPNMAC
REGISTRY["updet_mac"] = UPDETController
REGISTRY["ss_mac"] = SSMAC
REGISTRY["custom_mac"] = CustomMAC