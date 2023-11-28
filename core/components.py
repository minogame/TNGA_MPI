
import numpy as np, os, sys, glob, time, string, logging
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
np.set_printoptions(precision=4)
from time import gmtime, strftime
from random import choice, sample, choices

base_folder = './'
try:
    os.mkdir(base_folder+'log')
    os.mkdir(base_folder+'agent_pool')
    os.mkdir(base_folder+'job_pool')
    os.mkdir(base_folder+'result_pool')
except:
    pass


current_time = strftime("%Y%m%d_%H%M%S", gmtime())

log_name = 'sim_{}_{}_{}_a{}.log'.format('data', sys.argv[2], sys.argv[3], sys.argv[4])
logging.basicConfig(filename=log_name, filemode='a', level=logging.DEBUG,
                                        format='%(asctime)s: %(message)s', datefmt='%H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:  %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)



data = np.load('data.npz')
logging.info(data['adj_matrix'])
np.save('data.npy', data['goal'])
evoluation_goal = 'data.npy'



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