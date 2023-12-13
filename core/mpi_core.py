from dataclasses import dataclass

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
    assigned_job: None
    estimation_time: None
    current_iter: None
    tik_time: 0
    up_time: 0
    abnormal_counter: 0

@dataclass
class INDIVIDUAL_STATUS:
    individual: None
    assigned: False
    repeated: 0

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

