from mpi4py import MPI
from core.components import Overlord

class MPI_PROCESS:

    def __init__(self) -> None:
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

    def run_as_overlord(self) -> None:
        
        pass

    def run_as_agent(self) -> None:
        
        pass


if __name__ == '__main__':


if rank == 0:
    print(comm, rank)
    print(comm.Get_size())
    print(comm.Get_info())
    print(comm.Get_topology())
    data = {'a': 7, 'b': 3.14}
    comm.send(data, dest=1, tag=11)
elif rank == 1:
    data = comm.recv(source=0, tag=11)
    print(data)