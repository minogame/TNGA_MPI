from dataclasses import dataclass
from typing import Any

class TAGS:
    __rdict__ = {}
    DATA_ADJ_MATRIX = 0
    DATA_GOAL = 1
    DATA_RUN_REPORT = 2
    DATA_MISC = 3
    INFO_TIME_ESTIMATION = 10
    INFO_SURVIVAL = 11
    INFO_ABNORMAL = 12
    INFO_MISC = 12

class SURVIVAL:
    __rdict__ = {}
    HOST_RUNNING = 0
    HOST_NORMAL_FINISHED = 1
    HOST_ABNORMAL_SHUTDOWN = 2

class REASONS:
    __rdict__ = {}
    REACH_MAX_ITER = 0
    HARD_TIMEOUT = 1
    FAKE_RESULT = 2

@dataclass
class AGENT_STATUS:
    assigned_job: Any = None
    estimation_time: float = None
    current_iter: int = None
    tik_time: int = 0
    up_time: float = 0
    abnormal_counter: int = 0

@dataclass
class SOCIETY:
    name: None
    individuals: list[Any] = []
    score_original: list[Any] = []
    score_total: list[Any] = []
    indv_ranking: list[Any] = []

    def __iter__(self):
        for i in self.individuals:
            yield i.scope, i

    def __len__(self):
        return len(self.individuals)

@dataclass
class INDIVIDUAL_STATUS:
    individual: Any = None
    assigned: list[int] = []
    repeated: int = 0

    def __str__(self) -> str:
        if self.assigned:
            t = f'Individual {self.individual.scope} has been repeated {self.repeated} times,' \
                f'it is currently been assign in agent rank = {self.assigned}.'
            
        return t

class DUMMYINDV:
    pass

def DUMMYFUNC(*args, **kwds):
    pass

def init_rdict(c):
    for k, v in c.__dict__.items():
        if not k.startswith('__'):
            c.__rdict__[v] = k

init_rdict(TAGS)
init_rdict(REASONS)
init_rdict(SURVIVAL)

