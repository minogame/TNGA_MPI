import time, mpi4py
import numpy as np
import functools


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

    def __init__(self, comm: mpi4py.MPI.COMM_WORLD, **kwargs) -> None:
        self.kwargs = kwargs
        self.logger = kwargs['logger']


        # generation
        self.max_generation = kwargs['experiment']['max_generation']
        self.current_generation = None
        self.previous_generation = None
        self.N_generation = 0

        self.generation = kwargs['generation']
        self.generation_list = []


        self.available_agents = []
        self.known_agents = {}
        self.time = 0

    def tik(self, sec):
        self.time += sec
        time.sleep(sec)

    def call_with_interval(self, func, interval, **kwds):
        if self.time % interval == 0:
            func(**kwds)
        return

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

    def __report_agents__(self):
        self.logger.info('Current number of known agents is {}.'.format(len(self.known_agents)))
        self.logger.info(list(self.known_agents.keys()))

    def __report_generation__(self):
        self.logger.info('Current length of indv_to_distribute is {}.'.format(len(self.current_generation.indv_to_distribute)))
        self.logger.info('Current length of indv_to_collect is {}.'.format(len(self.current_generation.indv_to_collect)))
        self.logger.info([(indv.scope, indv.sge_job_id) for indv in self.current_generation.indv_to_collect])

    def __generation__(self):
        if self.N_generation > self.max_generation:
            return False
        else:
            if self.current_generation is None:
                self.current_generation = self.generation(name='generation_init', **self.kwargs)
                self.current_generation.indv_to_distribute = []

            if self.current_generation.is_finished():
                if self.previous_generation is not None:
                    self.current_generation(**self.kwargs)
                self.N_generation += 1
                self.previous_generation = self.current_generation
                self.current_generation = self.generation(self.previous_generation, 
                                                        name='generation_{:03d}'.format(self.N_generation), **self.kwargs)

            return True

    def __call__(self):
        while self.__generation__():
            self.__call_with_interval__(self.__check_available_agent__, 4)()
            self.__call_with_interval__(self.__assign_job__, 4)()
            self.__call_with_interval__(self.__collect_result__, 4)()
            self.__call_with_interval__(self.__report_agents__, 180)()
            self.__call_with_interval__(self.__report_generation__, 160)()
            self.__tik__(2)




def score_summary(obj):
    logging.info('===== {} ====='.format(obj.name))

    for k, v in obj.societies.items():
        logging.info('===== ISLAND {} ====='.format(k))

        for idx, indv in enumerate(v['indv']):
            if idx == v['rank'][0]:
                logging.info('\033[31m{} | {:.3f} | {} | {:.5f} | {}\033[0m'.format(indv.scope, np.log10(indv.sparsity), [ float('{:0.4f}'.format(l)) for l in indv.repeat_loss ], v['total'][idx], indv.parents))
                logging.info(indv.adj_matrix)
            else:
                logging.info('{} | {:.3f} | {} | {:.5f} | {}'.format(indv.scope, np.log10(indv.sparsity), [ float('{:0.4f}'.format(l)) for l in indv.repeat_loss ], v['total'][idx], indv.parents))
                logging.info(indv.adj_matrix)

if __name__ == '__main__':
    pipeline = Overlord(# GENERATION PROPERTIES
                        max_generation=30, generation=Generation, random_init=True,
                        # ISLAND PROPERTIES
                        N_islands=1, population=[int(sys.argv[2])], 
                        # INVIDUAL PROPERTIES
                        size=4, rank=2, out=2, init_sparsity=-0.00001,
                        # EVALUATION PROPERTIES
                        evaluate_repeat=2, max_iterations=10000,
                        fitness_func=[ lambda s,l: s+l*50],
                        #
                        # EVOLUTION PROPERTIES
                        elimiation_threshold=[int(sys.argv[3])], immigration_prob=0, immigration_number=5,
                        crossover_alpha=1, mutation_prob=0.05,
                        # FOR COMPUTATION
                        callbacks=[score_summary])
    pipeline()