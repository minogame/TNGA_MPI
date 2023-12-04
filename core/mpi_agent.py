from typing import Any
import numpy as np, os, sys, time, gc
np.set_printoptions(precision=2)
from tenmul4 import TensorNetwork
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import mpi4py
from mpi_core import TAGS

class MPI_Agent(object):

    def __init__(self, comm: mpi4py.MPI.COMM_WORLD, **kwargs) -> None:
        self.kwargs = kwargs
        self.comm = comm
        self.time = 0
        self.start_time = time.time()
        self.logger = kwargs['logger']
        self.optimizer = kwargs['optimization']['optimizer']
        self.optimizer_param = kwargs['optimization']['optimizer_params']

    def sync_goal(self):
        try:
            self.evoluation_goal = None
            self.evoluation_goal = self.comm.bcast(self.evoluation_goal, root=0)
        except:
            self.logger.error(self.rank, 'reported errors in receiving evoluation_goal')
            raise

    def tik(self, sec):
        self.time += sec
        time.sleep(sec)

    def report_surival(self, current_iter, max_iter):
        status, msg = self.req_surv.test()
        if status:
            real_up_time = time.time() - self.start_time
            return_dict = {
                'rank': self.rank,
                'time': self.time,
                'real_up_time': real_up_time,
                'busy': self.busy_status,
                'current_iter': current_iter,
                'max_iter': max_iter
            }
            self.comm.isend(return_dict, dest=0, tag=TAGS.INFO_SURVIVAL)
            self.logger.info(f'Received survival test singal {msg} from overload, ' \
                             f'reported tik time {self.time}, real up time {real_up_time},' \
                             f'current completion rate {current_iter} / {max_iter}.')
            self.req_surv = self.comm.irecv(tag=1)
        return

    def receive_job(self):
        status, msg = self.req_adjm.test()
        if status:
            try:
                indv_scope = msg['indv_scope']
                adj_matrix = msg['adj_matrix']
                max_iterations = msg['max_iterations']
            except:
                self.req_adjm.Cancel();self.req_adjm.Free()
                self.req_adjm = self.comm.irecv(source=0, tag=TAGS.DATA_ADJ_MATRIX)
                self.comm.isend(self.rank, dest=0, tag=TAGS.INFO_ABNORMAL)
        else:
            return None

        self.busy_status = True
        self.g = tf.Graph()
        self.sess = tf.compat.v1.Session(graph=self.g)

        with self.g.as_default():
            with tf.compat.v1.variable_scope(indv_scope):
                TN = TensorNetwork(adj_matrix)
                output = TN.reduction(random=False)
                goal = tf.convert_to_tensor(self.evoluation_goal)
                goal_square_norm = tf.convert_to_tensor(np.mean(np.square(self.evoluation_goal)))
                rse_loss = tf.reduce_mean(tf.square(output - goal)) / goal_square_norm
                step = TN.opt_opeartions(self.optimizer(**self.optimizer_param), rse_loss)
                var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=indv_scope)

        self.sess.run(tf.compat.v1.variables_initializer(var_list))

        return step, max_iterations, rse_loss

    def report_result(self, rse_loss, current_iter):

        loss = self.sess.run(rse_loss)
        self.sess.close()
        tf.reset_default_graph()
        del repeat_loss, g
        gc.collect()

        return_dict = {
            'rank': self.rank,
            'loss': loss,
            'current_iter': current_iter,
        }
        self.comm.isend(return_dict, dest=0, tag=TAGS.DATA_RUN_REPORT)
        self.busy_status = False
        return

    def evaluate(self, step, n_iter) -> None:
        for i in range(n_iter): 
            self.sess.run(step)
        return 

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        ## when the agent is called
        ## 1. initilize two mpi recv comm
        ## 2. sync goal with the rank = 0
        ## 3. entering the main while True loop
        ##      i. check if need to report surival state
        ##      ii. try to receive a job, initilized the tf sess and obtain a step
        ##      iii. run first N step, estimate overall run time, report to overload
        ##      iv. if estimate time is largely overhaul, then finish this run and report failure
        ##      v. run N step, try report surival state and count time
        ##      vi. if reach timeout or max step, stop run and exit while True loop
        ##      vii. report current steps and loss, clean tf sess and graph
        ##      viii. repeat from i. until the overload tell all things done
        ## 4. clean comm and report finish to rank = 0

        self.req_adjm = self.comm.irecv(source=0, tag=TAGS.DATA_ADJ_MATRIX)
        self.req_surv = self.comm.irecv(source=0, tag=TAGS.INFO_SURVIVAL)
        self.sync_goal()

        call_start_time = time.time()
        current_iter, job, step, max_iterations, rse_loss = None, None, None, None, None
        while True:
            self.report_surival(current_iter, max_iterations)

            if not self.busy_status:
                job = self.receive_job()
                if not job:
                    self.tik(1)
                    continue
                else:
                    step, max_iterations, rse_loss = job
                    self.busy_status = True
                    current_iter = 0

            
            if current_iter < max_iterations

            

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
        self.logger.info(f'Rank {self.rank} finished.')
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

                self.logger.info('Receiving individual {} for {}x{} ...'.format(scope, repeat, iters))

                try:
                    repeat_loss = evaluate(tf_graph=g, sess=sess, indv_scope=scope, adj_matrix=adj_matrix, evaluate_repeat=repeat, max_iterations=iters,
                                                                    evoluation_goal=evoluation_goal, evoluation_goal_square_norm=evoluation_goal_square_norm)
                    self.logger.info('Reporting result {}.'.format(repeat_loss))
                    np.savez(base_folder+'/result_pool/{}.npz'.format(scope.replace('/', '_')),
                                        repeat_loss=[ float('{:0.4f}'.format(l)) for l in repeat_loss ],
                                        adj_matrix=adj_matrix)

                    os.remove(base_folder+'/job_pool/{}.npz'.format(agent_id))
                    open(base_folder+'/agent_pool/{}.POOL'.format(agent_id), 'w').close()

                except Exception as e:
                    os.remove(base_folder+'/agent_pool/{}.POOL'.format(agent_id))
                    raise e


            time.sleep(1)
