from typing import Any
import numpy as np
import yaml
import argparse
from mpi4py import MPI
from core.components import Overlord
from core.mpi_agent import MPI_Agent
from core.mpi_overload import MPI_Overlord

class MPI_Process:

    def __init__(self, **kwds) -> None:
        self.config = kwds
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        if self.rank == 0:
            goal = kwds['experiment']['evoluation_goal']
            self.evoluation_goal = np.load(goal)
            self.evoluation_goal = self.comm.bcast(self.evoluation_goal, root=0)
            self.noumenon = MPI_Overlord(self.comm, **kwds)
        else:
            self.noumenon = MPI_Agent(self.comm, **kwds)

    def __call__(self,) -> Any:
        self.noumenon()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='The yaml config file.', default='configs/default.yaml')
    args = parser.parse_args()

    config = yaml.load(open(args.config), yaml.Loader)
    process = MPI_Process(config)
    process()