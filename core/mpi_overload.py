import time, mpi4py
import numpy as np
import functools, itertools
from mpi_generation import Generation, Individual
from mpi_core import TAGS, REASONS, DUMMYFUNC, AGENT_STATUS
from callbacks import CALLBACKS

class MPI_Overlord():

    ## OVERLORD SEND:
    ## 1. survival ping
    ## 2. job data

    ## OVERLORD RECEIVE
    ## 1. survival info
    ## 2. abnormal when receiving job
    ## 3. estimation info
    ## 4. job result report

    ## SINGLE JOB PIPELINE
    ## 1. overlord match an agent and an individual
    ## 2. overlord ask the individual to submit the data
    ## 3. overlord isend the data to agent
    ## 4. overlord receive result and report to the individual


    ################ UTILS ################
    def __init__(self, comm: mpi4py.MPI.COMM_WORLD, **kwds) -> None:
        self.kwds = kwds
        self.logger = kwds['logger']
        self.time = 0
        self.comm = comm
        self.agent_size = self.comm.Get_size() - 1

        # generation
        self.max_generation = kwds['experiment']['max_generation']
        self.collection_of_generations = []

        # agents
        self.available_agents = dict(
            itertools.zip_longest(list(range(1, self.agent_size+1)), [], fillvalue=AGENT_STATUS()))

    def tik_and_sleep(self, sec):
        self.time += sec
        time.sleep(sec)

    def call_with_interval(self, func, interval):
        if self.time % interval == 0:
            return func
        else:
            return DUMMYFUNC
        
    ################ MPI COMMUNICATION ################
    def sync_goal(self):
        goal = self.kwds['experiment']['evoluation_goal']
        self.evoluation_goal = np.load(goal)
        self.evoluation_goal = self.comm.bcast(self.evoluation_goal, root=0)

    def process_msg_surv(self, msg):
        rank_surv = msg['rank']

        self.available_agents[rank_surv].tik_time = msg['time']
        self.available_agents[rank_surv].up_time = msg['real_up_time']
        self.available_agents[rank_surv].current_iter = msg['current_iter']

        if msg['busy']:
            self.logger.info(f'Received survival report from agent rank {rank_surv}, ' \
                            f'reported tik time {msg["time"]}, real up time {msg["real_up_time"]},' \
                            f'current completion rate {msg["current_iter"]} / {msg["max_iter"]}.')
        else:
            self.logger.info(f'Received survival report from agent rank {rank_surv}, ' \
                            f'reported tik time {msg["time"]}, real up time {msg["real_up_time"]},' \
                            f'not working currently.')




    assigned_job: None
    estimation_time: None
    current_iter: None

    def call_agent_survivability(self):
        pass

    def tik_and_collect_everything_from_agent(self, sec):
        self.time += sec
        start_time = time.time()
        while time.time() - start_time < sec:

            status_surv, msg_surv = self.req_surv.test()
            if status_surv:
                msg_surv

                collected_queue.append(result)
                self.agent_state[source] = True
                print(f'Recv {result} from rank {source}, len(done)={len(collected_queue)}.')
                req_0 = self.comm.irecv(tag=0)

            status_estm, msg_estm = self.req_estm.test()
            if status:
                result, source = msg
                collected_queue.append(result)
                print(f'Recv special singal {result} from rank {source}.')
                req_1 = self.comm.irecv(tag=1)

            status_abnm, msg_abnm = self.req_abnm.test()
            if status:
                result, source = msg
                collected_queue.append(result)
                print(f'Recv special singal {result} from rank {source}.')
                req_1 = self.comm.irecv(tag=1)

            status_rept, msg_rept = self.req_rept.test()
            if status:
                result, source = msg
                collected_queue.append(result)
                print(f'Recv special singal {result} from rank {source}.')
                req_1 = self.comm.irecv(tag=1)

        return

    def __assign_job__(self):
        self.__check_available_agent__()
        if len(self.available_agents)>0:
            for agent in self.available_agents:
                self.current_generation.distribute_indv(agent)

    def __collect_result__(self):
        self.current_generation.collect_indv()

    def __report_generation__(self):
        self.logger.info('Current length of indv_to_distribute is {}.'.format(len(self.current_generation.indv_to_distribute)))
        self.logger.info('Current length of indv_to_collect is {}.'.format(len(self.current_generation.indv_to_collect)))
        self.logger.info([(indv.scope, indv.sge_job_id) for indv in self.current_generation.indv_to_collect])


    def __report_agents__(self):
        self.logger.info('Current number of known agents is {}.'.format(len(self.known_agents)))
        self.logger.info(list(self.known_agents.keys()))

    def span_generation(self):
        if not len(self.collection_of_generations):
            self.collection_of_generations.append(self.generation(name='generation_init', **self.kwds))
            return True

        if len(self.collection_of_generations) >= self.max_generation:
            return False
        
        else:
            ## is_finished now is TOTALLY finished, including evaluation and evolution
            current_generation = self.collection_of_generations[-1]
            if current_generation.is_finished():
                next_generation = self.generation(current_generation,
                    name=f'generation_{len(self.collection_of_generations)+1:03d}', **self.kwds)
                self.collection_of_generations.append(next_generation)

            return True

    def __call__(self):
        ## when the overload is called
        ## 1. initilize 4 mpi recv comm
        ## 2. sync goal with the rank = 0
        ## 3. entering the main generation spanning loop,
        ##    different from the former version,
        ##    now the overlord only send the messages from agent to generation,
        ##    the generation will deal with that.
        ## 4. clean comm and send finish msg to all the agents

        self.req_surv = self.comm.irecv(tag=TAGS.INFO_SURVIVAL)
        self.req_estm = self.comm.irecv(tag=TAGS.INFO_TIME_ESTIMATION)
        self.req_abnm = self.comm.irecv(tag=TAGS.INFO_ABNORMAL)
        self.req_rept = self.comm.irecv(tag=TAGS.DATA_RUN_REPORT)

        self.sync_goal()
        while self.span_generation():
            self.
            self.current_generation.indv_to_distribute = []
            self.call_with_interval(self.check_available_agent, 4)
            self.call_with_interval(self.__assign_job__, 4)
            self.call_with_interval(self.__collect_result__, 4)
            self.call_with_interval(self.__report_agents__, 180)
            self.call_with_interval(self.__report_generation__, 160)
            self.__tik__(2)

        else:
            CALLBACKS.OVERLOAD()

        self.kwds['overlord_propert']['collect_window']

if __name__ == '__main__':
    pipeline = Overlord
    pipeline()