from typing import Any
import numpy as np, os, sys, time, logging, gc
np.set_printoptions(precision=2)
from tenmul4 import TensorNetwork
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import mpi4py
from mpi_core import TAGS


class MPI_Agent(object):
    def __init__(self, comm, **kwargs):
        self.kwargs = kwargs
        self.comm = comm
        self.time = 0
        self.start_time = time.time()

    def sync_goal(self):
        try:
            self.evoluation_goal = None
            self.evoluation_goal = self.comm.bcast(self.evoluation_goal, root=0)
        except:
            print(self.rank, 'reported errors in receiving evoluation_goal')
            raise

    def report_surival(self):
        status, msg = self.req_surv.test()
        if status:
            real_up_time = time.time() - self.start_time
            self.comm.isend((self.rank, self.time, real_up_time), dest=0, tag=TAGS.INFO_SURVIVAL)
            print(f'Received survival test singal {msg} from overload, reported tik time {self.time}, real up time {real_up_time}.')
            self.req_surv = self.comm.irecv(tag=1)

    def receive_job(self):
        status, msg = self.req_surv.test()


        return None
        tf_graph, sess, indv_scope, adj_matrix, evaluate_repeat, max_iterations

        self.g = tf.Graph()
        self.sess = tf.Session(graph=self.g)

        with g.as_default():
            with tf.variable_scope(indv_scope):
                TN = TensorNetwork(adj_matrix)
                output = TN.reduction(random=False)
                goal = tf.convert_to_tensor(self.evoluation_goal)
                goal_square_norm = tf.convert_to_tensor(np.mean(np.square(self.evoluation_goal)))
                rse_loss = tf.reduce_mean(tf.square(output - goal)) / goal_square_norm
                step = TN.opt_opeartions(tf.train.AdamOptimizer(0.001), rse_loss)

        return step

    def report_result(self):


        self.sess.close()

        pass

    def evaluate(self):
        
        
        
        sess.close()
        tf.reset_default_graph()
        del repeat_loss, g
        gc.collect()

        


            repeat_loss = []
            for r in range(evaluate_repeat):
                sess.run(tf.compat.v1.variables_initializer(var_list))
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
        self.req_adjm = self.comm.irecv(source=0, tag=TAGS.DATA_ADJ_MATRIX)
        self.req_surv = self.comm.irecv(source=0, tag=TAGS.INFO_SURVIVAL)
        self.sync_goal()

        while True:
            report_surival()
            step = receive_job()
            if step:

            else:
                continue

            

            if status:
                if msg == "Done":
                    break
                else:
                    time.sleep(random.randint(1, 3))
                    self.comm.send((msg*2, self.rank), dest=0, tag=0)
                req = self.comm.irecv()
            
            if random.randint(0, 20) == 13:
                self.comm.isend(('AAAA', self.rank), dest=0, tag=1)
            time.sleep(1)


        self.req_adjm.Cancel();self.req_adjm.Free()
        self.req_surv.Cancel();self.req_surv.Free()
        print(f'Rank {self.rank} finished.')
        return



    
        while True:
            flag, evoluation_goal = check_and_load(agent_id)
            if flag:
                evoluation_goal_square_norm=
                indv = np.load(base_folder+'/job_pool/{}.npz'.format(agent_id))

                scope = indv['scope'].tolist()
                adj_matrix = indv['adj_matrix']
                repeat = indv['repeat']
                iters = indv['iters']

                logging.info('Receiving individual {} for {}x{} ...'.format(scope, repeat, iters))

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


            time.sleep(1)
