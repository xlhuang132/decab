import logging
import time
import os
from .utils import get_root_path
 
def create_logger(cfg):
    root_path=get_root_path(cfg)
    log_dir = os.path.join(root_path, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir,exist_ok=True)
    time_str = time.strftime("%Y-%m-%d-%H-%M")
    log_name = "{}.log".format(time_str)
    log_file = os.path.join(log_dir, log_name)
    # set up logger
    print("=> creating log {}".format(log_file))
    head = "%(asctime)-15s %(message)s"
    while len(logging.root.handlers)>0:
        logging.root.handlers.pop()
    logging.basicConfig(filename=str(log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    console = logging.StreamHandler()
    logging.getLogger("").addHandler(console)

    logger.info("---------------------Cfg is set as follow--------------------")
    logger.info(cfg)
    logger.info("-------------------------------------------------------------")
    return logger, log_file