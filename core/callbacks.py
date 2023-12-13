from typing import Any
from mpi_generation import Generation
import numpy as np
from pprint import pformat

class CALLBACKS:

    ### Callback functions are now automately called for each domain
    ### Callbacks are not called by initilization function,
    ### which calls all methods that defined (including do_nothing)

    class INDIVIDUAL:

        @staticmethod
        def do_nothing(*args, **kwds):
            pass

        def __init__(self, *args: Any, **kwds: Any) -> None:
            logger = kwds.get('kwds', None)
            for f in dir(self):
                if not f.startswith('__'):
                    ff = eval(f'self.{f}')
                    if logger:
                        logger.info(f'Calling callback function {ff}.')
                    ff(*args, **kwds)

    class GENERATION:

        @staticmethod
        def do_nothing(*args, **kwds):
            pass

        @staticmethod
        def score_summary(generation: Generation, logger):
            logger.info('===== {} ====='.format(generation.name))

            for k, v in generation.societies.items():
                logger.info('===== SOCIETY {} ====='.format(k))

                for idx, indv in enumerate(v['indv']):
                    if idx == v['rank'][0]:
                        logger.info('\033[31m{} | {:.3f} | {} | {:.5f} | {}\033[0m'.format(indv.scope, np.log10(indv.sparsity), [ float('{:0.4f}'.format(l)) for l in indv.repeat_loss ], v['total'][idx], indv.parents))
                        logger.info(indv.adj_matrix)
                    else:
                        logger.info('{} | {:.3f} | {} | {:.5f} | {}'.format(indv.scope, np.log10(indv.sparsity), [ float('{:0.4f}'.format(l)) for l in indv.repeat_loss ], v['total'][idx], indv.parents))
                        logger.info(indv.adj_matrix)

        def __init__(self, *args: Any, **kwds: Any) -> None:
            logger = kwds.get('kwds', None)
            for f in dir(self):
                if not f.startswith('__'):
                    ff = eval(f'self.{f}')
                    if logger:
                        logger.info(f'Calling callback function {ff}.')
                    ff(*args, **kwds)


    class OVERLOAD:

        @staticmethod
        def do_nothing(*args, **kwds):
            pass

        def __init__(self, *args: Any, **kwds: Any) -> None:
            logger = kwds.get('kwds', None)
            for f in dir(self):
                if not f.startswith('__'):
                    ff = eval(f'self.{f}')
                    if logger:
                        logger.info(f'Calling callback function {ff}')
                    ff(*args, **kwds)