from typing import Any
import yaml
import argparse
from mpi4py import MPI
from core.mpi_agent import MPI_Agent
from core.mpi_overload import MPI_Overlord
import logging
logging.basicConfig(format='[%(levelname)s] +%(asctime)s+ =%(name)s= %(message)s ', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)

class MPI_Process:

    def __init__(self, **kwds) -> None:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        logger = logging.getLogger(f'logger_rank_{rank}')

        if rank == 0:
            self.noumenon = MPI_Overlord(comm, logger=logger, **kwds)
        else:
            self.noumenon = MPI_Agent(comm, logger=logger, **kwds)

    def __call__(self,) -> Any:
        self.noumenon()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='The yaml config file.', default='configs/default.yaml')
    args = parser.parse_args()

    config = yaml.load(open(args.config), yaml.Loader)
    MPI_Process(config)()