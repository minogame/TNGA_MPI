import numpy as np, os, sys, time, logging, gc
np.set_printoptions(precision=2)
from tenmul4 import TensorNetwork
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from pathlib import Path

agent_id = sys.argv[1]
base_folder = './'
try:
    os.mkdir(base_folder+'log')
    os.mkdir(base_folder+'agent_pool')
    os.mkdir(base_folder+'job_pool')
    os.mkdir(base_folder+'result_pool')
except:
    pass

log_name = base_folder + '/log/{}.log'.format(agent_id)
logging.basicConfig(filename=log_name, filemode='a', level=logging.DEBUG,
                                        format='%(asctime)s: %(message)s', datefmt='%H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:  %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

if __name__ == '__main__':
    Path(base_folder+'/agent_pool/{}.POOL'.format(agent_id)).touch()
