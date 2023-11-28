import time




class MPI_Overlord():

    def __init__(self, max_generation=100, **kwargs):
        self.dummy_func = lambda *args, **kwargs: None
        self.max_generation = max_generation
        self.current_generation = None
        self.previous_generation = None
        self.N_generation = 0
        self.kwargs = kwargs
        self.generation = kwargs['generation']
        self.generation_list = []
        self.available_agents = []
        self.known_agents = {}
        self.time = 0

    def __call_with_interval__(self, func, interval):
        return func if self.time%interval == 0 else self.dummy_func

    def __tik__(self, sec):
        # logging.info(self.time)
        self.time += sec
        time.sleep(sec)

    def __check_available_agent__(self):
        self.available_agents.clear()
        agents = glob.glob(base_folder+'/agent_pool/*.POOL')
        agents_id = [ a.split('/')[-1][:-5] for a in agents ]

        for aid in list(self.known_agents.keys()):
            if aid not in agents_id:
                logging.info('Dead agent id = {} found!'.format(aid))
                self.known_agents.pop(aid, None)

        for aid in agents_id:
            if aid in self.known_agents.keys():
                if self.known_agents[aid].is_available():
                    self.available_agents.append(self.known_agents[aid])
            else:
                self.known_agents[aid] = Agent(sge_job_id=aid)
                logging.info('New agent id = {} found!'.format(aid))

    def __assign_job__(self):
        self.__check_available_agent__()
        if len(self.available_agents)>0:
            for agent in self.available_agents:
                self.current_generation.distribute_indv(agent)

    def __collect_result__(self):
        self.current_generation.collect_indv()

    def __report_agents__(self):
        logging.info('Current number of known agents is {}.'.format(len(self.known_agents)))
        logging.info(list(self.known_agents.keys()))

    def __report_generation__(self):
        logging.info('Current length of indv_to_distribute is {}.'.format(len(self.current_generation.indv_to_distribute)))
        logging.info('Current length of indv_to_collect is {}.'.format(len(self.current_generation.indv_to_collect)))
        logging.info([(indv.scope, indv.sge_job_id) for indv in self.current_generation.indv_to_collect])

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