from mpi_generation import Generation
import numpy as np

class CALLBACKS:

    ### Callbacks are now automately called for each domain

    class GENERATION:

        @staticmethod
        def score_summary(generation: Generation, logger):
            logger.info('===== {} ====='.format(generation.name))

            for k, v in generation.societies.items():
                logger.info('===== ISLAND {} ====='.format(k))

                for idx, indv in enumerate(v['indv']):
                    if idx == v['rank'][0]:
                        logger.info('\033[31m{} | {:.3f} | {} | {:.5f} | {}\033[0m'.format(indv.scope, np.log10(indv.sparsity), [ float('{:0.4f}'.format(l)) for l in indv.repeat_loss ], v['total'][idx], indv.parents))
                        logger.info(indv.adj_matrix)
                    else:
                        logger.info('{} | {:.3f} | {} | {:.5f} | {}'.format(indv.scope, np.log10(indv.sparsity), [ float('{:0.4f}'.format(l)) for l in indv.repeat_loss ], v['total'][idx], indv.parents))
                        logger.info(indv.adj_matrix)

    class INDIVIDUAL:

        @staticmethod
        def do_nothing(*args, **kwds):
            pass

    class OVERLOAD:

        @staticmethod
        def do_nothing(*args, **kwds):
            pass