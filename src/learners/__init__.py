from .nq_learner import NQLearner
from .nq_role_learner import NQLearner as NQRoleLearner

REGISTRY = {}

REGISTRY["nq_learner"] = NQLearner
REGISTRY["nq_role_learner"] = NQRoleLearner