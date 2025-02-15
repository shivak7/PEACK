import logging
from datetime import datetime

def start_logger():

    log_path = '../../logs/'
    fname = 'PEACK_' + datetime.today().strftime('%Y-%m-%d') + '.log'
    logger = logging.getLogger('PEACK')

    fh = logging.FileHandler(fname)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)


print('Initializing PEACK...')
start_logger()

