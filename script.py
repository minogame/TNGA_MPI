import random
from typing import Any
from mpi4py import MPI
import time

class MPI_PROCESS:

    def __init__(self) -> None:
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        if self.rank == 0:
            self.queue = list(range(100))

    def run_as_overlord(self) -> None:
        collected_queue = []
        self.agent_state = dict()
        self.others = list(range(1, self.size))
        req_0 = self.comm.irecv(tag=0)
        req_1 = self.comm.irecv(tag=1)
        for r in self.others:
            self.agent_state[r] = True
        
        while True:
            if self.queue:
                for r in self.others:
                    if self.agent_state[r]:
                        d = self.queue.pop(0)
                        self.comm.isend(d, dest=r)
                        print(f'Send {d} to rank {r}.')
                        self.agent_state[r] = False

            status, msg = req_0.test()
            if status:
                result, source = msg
                collected_queue.append(result)
                self.agent_state[source] = True
                print(f'Recv {result} from rank {source}, len(done)={len(collected_queue)}.')
                req_0 = self.comm.irecv(tag=0)

            status, msg = req_1.test()
            if status:
                result, source = msg
                collected_queue.append(result)
                print(f'Recv special singal {result} from rank {source}.')
                req_1 = self.comm.irecv(tag=1)

            if len(collected_queue) == 100:
                break

            time.sleep(1)
        req_0.Cancel()
        req_0.Free()
        req_1.Cancel()
        req_1.Free()

        for r in self.others:
            self.comm.isend('Done', dest=r)

        print(collected_queue)

        return

    def run_as_agent(self) -> None:
        req = self.comm.irecv()
        while True:
            status, msg = req.test()

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

        print(f'Rank {self.rank} finished.')
        return
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if self.rank == 0:
            self.run_as_overlord()
        else:
            self.run_as_agent()

        return
    
if __name__ == '__main__':
    x = MPI_PROCESS()
    x()