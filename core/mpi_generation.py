import numpy as np
import random, string, os


class Individual(object):

    def __init__(self, adj_matrix=None, scope=None, **kwargs):
        super(Individual, self).__init__()
        if adj_matrix is None:
            self.adj_matrix = kwargs['adj_func'](**kwargs)
        else:
            self.adj_matrix = adj_matrix
        self.scope = scope
        self.parents = kwargs['parents'] if 'parents' in kwargs.keys() else None
        self.repeat = kwargs['evaluate_repeat'] if 'evaluate_repeat' in kwargs.keys() else 1
        self.iters = kwargs['max_iterations'] if 'max_iterations' in kwargs.keys() else 10000
        self.dim = self.adj_matrix.shape[0]
        self.adj_matrix[np.tril_indices(self.dim, -1)] = self.adj_matrix.transpose()[np.tril_indices(self.dim, -1)]
        adj_matrix_k = np.copy(self.adj_matrix)
        adj_matrix_k[adj_matrix_k==0] = 1
        self.present_elements = np.prod(np.diag(adj_matrix_k))
        self.actual_elements = np.sum([ np.prod(adj_matrix_k[d]) for d in range(self.dim) ])
        self.sparsity = self.actual_elements/self.present_elements
        self.sparsity_connection = np.sum(self.adj_matrix[np.triu_indices(self.adj_matrix.shape[0], 1)]>0)

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
    class DummyIndv: pass

    def __init__(self, pG=None, name=None, **kwargs):
        super(Generation, self).__init__()
        self.name = name
        self.N_islands = kwargs['N_islands'] if 'N_islands' in kwargs.keys() else 1
        self.kwargs = kwargs
        self.out = self.kwargs['out']
        self.rank = self.kwargs['rank']
        self.size = self.kwargs['size']
        self.init_sparsity = kwargs['init_sparsity'] if 'init_sparsity' in kwargs.keys() else 0.8
        self.indv_to_collect = []
        self.indv_to_distribute = []
        if pG is not None:
            self.societies = {}
            for k, v in pG.societies.items():
                self.societies[k] = {}
                self.societies[k]['indv'] = \
                        [ Individual(adj_matrix=indv.adj_matrix, parents=indv.parents,
                          scope='{}/{}/{:03d}'.format(self.name, k, idx), **self.kwargs) \
                        for idx, indv in enumerate(v['indv']) ]
                self.indv_to_distribute += [indv for indv in self.societies[k]['indv']]

        elif 'random_init' in kwargs.keys():
            self.societies = {}
            for n in range(self.kwargs['N_islands']):
                society_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
                self.societies[society_name] = {}
                self.societies[society_name]['indv'] = [ \
                        Individual(scope='{}/{}/{:03d}'.format(self.name, society_name, i), 
                        adj_func=self.__random_adj_matrix__, **self.kwargs) \
                        for i in range(self.kwargs['population'][n]) ]
                self.indv_to_distribute += [indv for indv in self.societies[society_name]['indv']]

    def __call__(self, **kwargs):
        try:
            self.__evaluate__()
            if 'callbacks' in kwargs.keys():
                for c in kwargs['callbacks']:
                    c(self)
            self.__evolve__()
            return True
        except Exception as e:
            raise e

    def __random_adj_matrix__(self, **kwargs):
        if isinstance(self.out, list):
            adj_matrix = np.diag(self.out)
        else:
            adj_matrix = np.diag([self.out]*self.size)

        if self.init_sparsity < 0:
            connection = []
            real_init_sparsity = np.random.uniform(low=-self.init_sparsity, high=1.0)
            for i in range(np.sum(np.arange(self.size))):
                connection.append(int(np.random.uniform()>real_init_sparsity)*self.rank)
        else:
            connection = [ int(np.random.uniform()>self.init_sparsity)*self.rank for i in range(np.sum(np.arange(self.size)))]
        adj_matrix[np.triu_indices(self.size, 1)] = connection
        return adj_matrix

    def __evolve__(self):
        def mutation(indv, prob):
            dim = indv.adj_matrix.shape[0]
            elements = np.stack(np.triu_indices(dim, 1)).transpose()
            mask = np.random.uniform(size=elements.shape[0])<prob
            mutated_elements = tuple(map(tuple, elements[mask].transpose()))
            if mutated_elements:
                indv.adj_matrix[mutated_elements] = self.rank - indv.adj_matrix[mutated_elements]
                indv.adj_matrix[np.tril_indices(dim, -1)] = indv.adj_matrix.transpose()[np.tril_indices(dim, -1)]

        def immigration(islands, number=5):
            island_A, island_B = islands
            self.logger.info('immigration happend!')
            for _ in range(number):
                island_B.append(island_A.pop(0))
                island_A.append(island_B.pop(0))

        def elimination(island, threshold=80):
            island['rank'] = island['rank'][:threshold]
            island['indv'] = [island['indv'][i] for i in island['rank']]
            island['total'] = [island['total'][i] for i in island['rank']]

        def crossover(island, population, alpha=5):
            __adj_matrix__, __parents__ = [], []
            def propagation(couple, percent=0.5):
                adj_matrix_male = np.copy(couple[0].adj_matrix)
                adj_matrix_female = np.copy(couple[1].adj_matrix)

                dim = adj_matrix_male.shape[0]
                exchange_core = choice(list(range(dim)))

                exchange = adj_matrix_male[exchange_core]
                adj_matrix_male[exchange_core] = adj_matrix_female[exchange_core] 
                adj_matrix_female[exchange_core] = exchange

                adj_matrix_male[np.tril_indices(dim, -1)] = adj_matrix_male.transpose()[np.tril_indices(dim, -1)]
                adj_matrix_female[np.tril_indices(dim, -1)] = adj_matrix_female.transpose()[np.tril_indices(dim, -1)]

                __adj_matrix__.append(adj_matrix_male)
                __adj_matrix__.append(adj_matrix_female)
                __parents__.append((couple[0].scope[-13:], couple[1].scope[-13:]))
                __parents__.append((couple[0].scope[-13:], couple[1].scope[-13:]))

            indv, fitness = island['indv'], island['total']
            rank = np.argsort(fitness)
            # prob = [ 1.0/(1e-5+f)*alpha for f in fitness]        
            # p = [ np.exp(3/(1+k)) for k in range(len(indv)) ]
            p = [ np.maximum(np.log(float(sys.argv[4])/(0.01+k*5)), 0.01) for k in range(population) ]
            prob = np.zeros(len(indv))
            for idx, i in enumerate(rank): prob[i] = p[idx]
            for i in range(population//2): propagation(choices(indv, weights=prob, k=2))
            for i in range(population-len(indv)): indv.append(DummyIndv())
            for v, m, p in zip(indv, __adj_matrix__, __parents__): v.adj_matrix, v.parents = m, p

        # ELIMINATION
        if 'elimiation_threshold' in self.kwargs:
            for idx, (k, v) in enumerate(self.societies.items()):
                elimination(v, self.kwargs['elimiation_threshold'][idx])

        # IMMIRATION
        if 'immigration_prob' in self.kwargs:
            if np.random.uniform()<self.kwargs['immigration_prob']:
                immigration(sample([v['indv'] for k, v in self.societies.items()], k=2), self.kwargs['immigration_number'])

        # CROSSOVER
        if 'crossover_alpha' in self.kwargs:
            for idx, (k, v) in enumerate(self.societies.items()):
                crossover(v, self.kwargs['population'][idx], self.kwargs['crossover_alpha'])

        # MUTATION
        if 'mutation_prob' in self.kwargs:
            for k, v in self.societies.items():
                for indv in v['indv']:
                    mutation(indv, self.kwargs['mutation_prob'])

    def __evaluate__(self):

        def score2rank(island, idx):
            score = island['score']
            sparsity_score = [ s for s, _ in score ]
            loss_score = [ l for _, l in score ]

            if 'fitness_func' in self.kwargs.keys():
                if isinstance(self.kwargs['fitness_func'], list):
                    fitness_func = self.kwargs['fitness_func'][idx]
                else:
                    fitness_func = self.kwargs['fitness_func']
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
        if len(self.indv_to_distribute) == 0 and len(self.indv_to_collect) == 0:
            return True
        else:
            return False