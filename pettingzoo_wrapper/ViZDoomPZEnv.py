from pettingzoo import ParallelEnv
from pettingzoo.utils.env import ObsDict, ActionDict, AgentID
from .multi_agent_env import MultiAgentViZDoomEnv

class ViZDoomPZEnv(ParallelEnv):
    def __init__(self, level, render_mode):
        pass