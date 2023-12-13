from typing import Any, Callable
import numpy as np
import random, string, os
from evolve import EVOLVE_OPS
from callbacks import CALLBACKS


class Individual(object):

    ## Creation of adj matrix based on function now is method of individual.
    ## functions now is called by Generation by staticmethod@individual

    @staticmethod
    def full_connection_adj_matrix(individual):
        if isinstance(individual.presented_shape, list):
            adj_matrix = np.diag(individual.presented_shape)
        else:
            adj_matrix = np.diag([individual.presented_shape]*individual.tn_size)

        adj_matrix[np.triu_indices(individual.tn_size, 1)] = individual.tn_rank

        return adj_matrix

    @staticmethod
    def naive_random_adj_matrix_with_sparsity_limitation(individual):
        if isinstance(individual.presented_shape, list):
            adj_matrix = np.diag(individual.presented_shape)
        else:
            adj_matrix = np.diag([individual.presented_shape]*individual.tn_size)

        if individual.init_sparsity < 0:
            connection = []
            real_init_sparsity = np.random.uniform(low=-individual.init_sparsity, high=1.0)
            for _ in range(np.sum(np.arange(individual.tn_size))):
                connection.append(int(np.random.uniform()>real_init_sparsity)*individual.tn_rank)
        else:
            connection = [ int(np.random.uniform()>individual.init_sparsity)*individual.tn_rank for _ in range(np.sum(np.arange(individual.tn_size)))]
        adj_matrix[np.triu_indices(individual.tn_size, 1)] = connection
        return adj_matrix

    def __init__(self, adj_matrix=None, scope=None, **kwds):
        super(Individual, self).__init__()

        ## basic propoerties
        if adj_matrix is None:
            self.adj_matrix = kwds['adj_func'](**kwds)
        elif adj_matrix is Callable:
            self.adj_matrix = adj_matrix(self)
        else:
            self.adj_matrix = adj_matrix

        self.adj_matrix[np.tril_indices(self.dim, -1)] = self.adj_matrix.transpose()[np.tril_indices(self.dim, -1)]

        self.scope = scope # this is also the "name" of the individual
        self.dim = self.adj_matrix.shape[0]
        self.repeat_loss = []

        ## parse the kwds
        self.parents = kwds.get('parents', None)
        self.individual_property = kwds.get('individual_property', {})
        self.random_initilization_property = self.individual_property.get('random_initilization_property', {})

        ## for random initilization
        self.tn_size = self.random_initilization_property.get('tn_size', 4)
        self.tn_rank = self.random_initilization_property.get('tn_rank', 2)
        self.presented_shape = self.random_initilization_property.get('presented_shape', 2)
        self.init_sparsity = self.random_initilization_property.get('init_sparsity', -0.00001) 

        ## adj_matrix_k put all the 0 edge to 1, this is only used for calulation of sparsity
        ## sparsity is the ratio of actual # elements to its presented # elements
        ## sparsity_connection is the # connections (compared to full connection)
        adj_matrix_k = np.copy(self.adj_matrix)
        adj_matrix_k[adj_matrix_k==0] = 1
        self.present_elements = np.prod(np.diag(adj_matrix_k))
        self.actual_elements = np.sum([ np.prod(adj_matrix_k[d]) for d in range(self.dim) ])
        self.sparsity = self.actual_elements / self.present_elements
        self.sparsity_connection = np.sum(self.adj_matrix[np.triu_indices(self.adj_matrix.shape[0], 1)] > 0)

    def __call__(self, action=None, *args: Any, **kwds: Any) -> Any:
        ## call an individual act as follows
        ## 1. depoly: report its adj_matrix to generation, 
        ##            generation will then forward it to overlord,
        ##            individual then tracks the rank it passed to.
        ## 2. collect: overlord report the repeat_loss to generation, 
        ##             generation forward this loss to individual,
        ##             individual process if the loss (discard of keep) then append it to repeat loss.

        if action == 'deploy':
            return dict(adj_matrix=self.adj_matrix, scope=self.scope)
        
        elif action == 'collect':
            reported_result = kwds.get('reported_result', None)
            pass
        else:
            pass

        return


    def deploy(self, sge_job_id):
        try:
            path = base_folder+'/job_pool/{}.npz'.format(sge_job_id)
            np.savez(path, adj_matrix=self.adj_matrix, scope=self.scope, repeat=self.repeat, iters=self.iters)
            self.sge_job_id = sge_job_id
            return True
        except Exception as e:
            raise e

    def collect(self, fake_loss=False):
        if not fake_loss:
            try:
                path = base_folder+'/result_pool/{}.npz'.format(self.scope.replace('/', '_'))
                result = np.load(path)
                self.repeat_loss = result['repeat_loss']
                os.remove(path)
                return True
            except Exception:
                return False
        else:
            self.repeat_loss = [9999]*self.repeat
            return True      

class Generation(object):

    ## When generation is called:
    ## 1. check is_finished.
    ## 2. if finished, do evaluate and evolve, then report.
    ## 3. if not finished, report the schedule table.

    def __init__(self, pG=None, name=None, **kwds):
        super(Generation, self).__init__()
        self.name = name
        self.N_islands = kwds['N_islands'] if 'N_islands' in kwds.keys() else 1
        self.kwds = kwds
        self.presented_shape = self.kwds['presented_shape']
        self.tn_rank = self.kwds['rank']
        self.tn_size = self.kwds['size']
        self.init_sparsity = kwds['init_sparsity'] if 'init_sparsity' in kwds.keys() else 0.8
        self.indv_to_collect = []
        self.indv_to_distribute = []
        if pG is not None:
            self.societies = {}
            for k, v in pG.societies.items():
                self.societies[k] = {}
                self.societies[k]['indv'] = \
                        [ Individual(adj_matrix=indv.adj_matrix, parents=indv.parents,
                          scope='{}/{}/{:03d}'.format(self.name, k, idx), **self.kwds) \
                        for idx, indv in enumerate(v['indv']) ]
                self.indv_to_distribute += [indv for indv in self.societies[k]['indv']]

        elif 'random_init' in kwds.keys():
            self.societies = {}
            for n in range(self.kwds['N_islands']):
                society_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
                self.societies[society_name] = {}
                self.societies[society_name]['indv'] = [ \
                        Individual(scope='{}/{}/{:03d}'.format(self.name, society_name, i), 
                        adj_func=self.__random_adj_matrix__, **self.kwds) \
                        for i in range(self.kwds['population'][n]) ]
                self.indv_to_distribute += [indv for indv in self.societies[society_name]['indv']]

    def __call__(self, **kwds):
        try:
            self.__evaluate__()
            if 'callbacks' in kwds.keys():
                for c in kwds['callbacks']:
                    c(self)
            self.__evolve__()
            return True
        except Exception as e:
            raise e

    def __evolve__(self):
        # ELIMINATION
        if 'elimiation_threshold' in self.kwds:
            for idx, (k, v) in enumerate(self.societies.items()):
                EVOLVE_OPS.elimination(v, self.kwds['elimiation_threshold'][idx])

        # IMMIRATION
        if 'immigration_prob' in self.kwds:
            if np.random.uniform()<self.kwds['immigration_prob']:
                EVOLVE_OPS.immigration(random.sample([v['indv'] for k, v in self.societies.items()], k=2), self.kwds['immigration_number'])

        # CROSSOVER
        if 'crossover_alpha' in self.kwds:
            for idx, (k, v) in enumerate(self.societies.items()):
                EVOLVE_OPS.crossover(v, self.kwds['population'][idx], self.kwds['crossover_alpha'])

        # MUTATION
        if 'mutation_prob' in self.kwds:
            for k, v in self.societies.items():
                for indv in v['indv']:
                    EVOLVE_OPS.mutation(indv, self.kwds['mutation_prob'])

    def __evaluate__(self):

        def score2rank(island, idx):
            score = island['score']
            sparsity_score = [ s for s, _ in score ]
            loss_score = [ l for _, l in score ]

            if 'fitness_func' in self.kwds.keys():
                if isinstance(self.kwds['fitness_func'], list):
                    fitness_func = self.kwds['fitness_func'][idx]
                else:
                    fitness_func = self.kwds['fitness_func']
            else:
                fitness_func = lambda s, l: s+100*l

            total_score = [ fitness_func(s, l) for s, l in zip(sparsity_score, loss_score) ]

            island['rank'] = np.argsort(total_score)
            island['total'] = total_score

        # RANKING
        for idx, (k, v) in enumerate(self.societies.items()):
            v['score'] = [ (indv.sparsity ,np.min(indv.repeat_loss)) for indv in v['indv'] ]
            score2rank(v, idx)

    def distribute_indv(self, agent):
        if self.indv_to_distribute:
            indv = self.indv_to_distribute.pop(0)
            if np.log10(indv.sparsity)<1.0:
                agent.receive(indv)
                self.indv_to_collect.append(indv)
                self.logger.info('Assigned individual {} to agent {}.'.format(indv.scope, agent.sge_job_id))
            else:
                indv.collect(fake_loss=True)
                self.logger.info('Individual {} is killed due to its sparsity = {} / {}.'.format(indv.scope, np.log10(indv.sparsity), indv.sparsity_connection))

    def collect_indv(self):
        for indv in self.indv_to_collect:
            if indv.collect():
                self.logger.info('Collected individual result {}.'.format(indv.scope))
                self.indv_to_collect.remove(indv)

    def is_finished(self):

        if self.previous_generation is not None:
        self.current_generation(**self.kwds)
        CALLBACKS.GENERATION(generation=self.current_generation)
    
        if len(self.indv_to_distribute) == 0 and len(self.indv_to_collect) == 0:
            return True
        else:
            return False