import time, mpi4py
import numpy as np
import functools
from mpi_generation import Generation, Individual
from mpi_core import TAGS, REASONS, DUMMYFUNC
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

        # generation
        self.max_generation = kwds['experiment']['max_generation']
        self.collection_of_generations = []

        # agents
        self.available_agents = []
        self.known_agents = {}

    def tik(self, sec):
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

    def check_agent_availablity(self):
        self.available_agents.clear()
        agents = glob.glob(base_folder+'/agent_pool/*.POOL')
        agents_id = [ a.split('/')[-1][:-5] for a in agents ]

        for aid in list(self.known_agents.keys()):
            if aid not in agents_id:
                self.logger.info('Dead agent id = {} found!'.format(aid))
                self.known_agents.pop(aid, None)

        for aid in agents_id:
            if aid in self.known_agents.keys():
                if self.known_agents[aid].is_available():
                    self.available_agents.append(self.known_agents[aid])
            else:
                self.known_agents[aid] = Agent(sge_job_id=aid)
                self.logger.info('New agent id = {} found!'.format(aid))

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
        ## 3. entering the main generation spanning loop
        ## 4. clean comm and send finish msg to all the agents

    ## 1. survival info
    ## 2. abnormal when receiving job
    ## 3. estimation info
    ## 4. job result report

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

        CALLBACKS.OVERLOAD

if __name__ == '__main__':
    pipeline = Overlord
    pipeline()