import inspect

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

def init_rdict(c):
    for k, v in c.__dict__.items():
        if not k.startswith('__'):
            c.__rdict__[v] = k

init_rdict(TAGS)
init_rdict(REASONS)


