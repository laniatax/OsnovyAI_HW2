import logging

def setup_logger(log_file='training.log'):
    logger = logging.getLogger('training_logger')
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        # Формат вывода
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Потоковый обработчик (консоль)
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # Файловый обработчик
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger
