import time
from common import *
from beratools.third_party.SegFormer_LineFootprint_Metaflow_preds import SeismicLinePredictionFlow


def ai_line_detection(callback, in_chm, in_model, patch_size, overlap_size, out_dir, processes, verbose):
    flow = SeismicLinePredictionFlow()
    flow.start()
    callback('Line detection done.')


if __name__ == '__main__':
    in_args, in_verbose = check_arguments()
    start_time = time.time()
    ai_line_detection(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)
    print('Elapsed time: {}'.format(time.time() - start_time))
