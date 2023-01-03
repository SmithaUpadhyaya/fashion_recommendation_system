from datetime import datetime
import logging
import os

def write_log(message, level = logging.INFO):
    
    curr_dir = os.getcwd()
    logs_path = os.path.join(curr_dir, "logs", "log_files")
    
    while os.path.exists(logs_path) == False:

        curr_dir = os.path.join(curr_dir, os.pardir)
        logs_path = os.path.join(curr_dir, "logs", "log_files")


    #LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
    LOG_FILE = f"app_log_{datetime.now().strftime('%m_%d_%Y')}.log"

    LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

    logging.basicConfig(
        filename = LOG_FILE_PATH,
        format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
        level = logging.DEBUG,
    )

    #os.makedirs(logs_path, exist_ok = True)

    if level == logging.DEBUG:
        logging.debug(message)

    elif level == logging.ERROR:
        logging.error(message)

    elif level == logging.INFO:
        logging.info(message)

    return
