from typing import Any
import numpy as np, os, sys, time, logging, gc
np.set_printoptions(precision=2)
from tenmul4 import TensorNetwork
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import mpi4py


class MPI_Agent(object):
    def __init__(self, comm, **kwargs):
        self.kwargs = kwargs
        self.comm = comm

    def receive(self, indv):
        indv.deploy(self.sge_job_id)
        with open(base_folder+'/agent_pool/{}.POOL'.format(self.sge_job_id), 'a') as f:
            f.write(evoluation_goal)

    def is_available(self):
        return True if os.stat(base_folder+'/agent_pool/{}.POOL'.format(self.kwargs['sge_job_id'])).st_size == 0 else False


    def evaluate(tf_graph, sess, indv_scope, adj_matrix, evaluate_repeat, max_iterations, evoluation_goal=None, evoluation_goal_square_norm=None):        
        with tf_graph.as_default():
            with tf.variable_scope(indv_scope):
                TN = TensorNetwork(adj_matrix)
                output = TN.reduction(False)
                goal = tf.convert_to_tensor(evoluation_goal)
                goal_square_norm = tf.convert_to_tensor(evoluation_goal_square_norm)
                rse_loss = tf.reduce_mean(tf.square(output - goal)) / goal_square_norm
                var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=indv_scope)
                step = tf.train.AdamOptimizer(0.001).minimize(rse_loss, var_list=var_list)
                var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=indv_scope)

            repeat_loss = []
            for r in range(evaluate_repeat):
                sess.run(tf.variables_initializer(var_list))
                for i in range(max_iterations): 
                    sess.run(step)
                repeat_loss.append(sess.run(rse_loss))

        return repeat_loss

    def check_and_load(agent_id):
        file_name = base_folder+'/agent_pool/{}.POOL'.format(agent_id)
        if os.stat(file_name).st_size == 0:
            return False, False
        else:
            with open(file_name, 'r') as f:
                goal_name = f.readline()
                evoluation_goal = np.load(goal_name).astype(np.float32)
            return True, evoluation_goal
        

    def __call__(self, *args: Any, **kwds: Any) -> Any:
    
        while True:
            flag, evoluation_goal = check_and_load(agent_id)
            if flag:
                evoluation_goal_square_norm=np.mean(np.square(evoluation_goal))
                indv = np.load(base_folder+'/job_pool/{}.npz'.format(agent_id))

                scope = indv['scope'].tolist()
                adj_matrix = indv['adj_matrix']
                repeat = indv['repeat']
                iters = indv['iters']

                logging.info('Receiving individual {} for {}x{} ...'.format(scope, repeat, iters))

                g = tf.Graph()
                sess = tf.Session(graph=g)
                try:
                    repeat_loss = evaluate(tf_graph=g, sess=sess, indv_scope=scope, adj_matrix=adj_matrix, evaluate_repeat=repeat, max_iterations=iters,
                                                                    evoluation_goal=evoluation_goal, evoluation_goal_square_norm=evoluation_goal_square_norm)
                    logging.info('Reporting result {}.'.format(repeat_loss))
                    np.savez(base_folder+'/result_pool/{}.npz'.format(scope.replace('/', '_')),
                                        repeat_loss=[ float('{:0.4f}'.format(l)) for l in repeat_loss ],
                                        adj_matrix=adj_matrix)

                    os.remove(base_folder+'/job_pool/{}.npz'.format(agent_id))
                    open(base_folder+'/agent_pool/{}.POOL'.format(agent_id), 'w').close()

                except Exception as e:
                    os.remove(base_folder+'/agent_pool/{}.POOL'.format(agent_id))
                    raise e

                sess.close()
                tf.reset_default_graph()
                del repeat_loss, g
                gc.collect()

            time.sleep(1)
