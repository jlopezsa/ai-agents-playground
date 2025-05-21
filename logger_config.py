import logging
from colorlog import ColoredFormatter

def setup_logger():
    # Configura el logger raíz
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Elimina handlers existentes
    if logger.hasHandlers():
        logger.handlers.clear()

    # Crea un handler de consola
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)

    # Define un formatter con colores
    formatter = ColoredFormatter(
        '%(asctime)s - %(log_color)s%(levelname)s%(reset)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'bold_red',
        }
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Silencia los logs INFO/DEBUG de LangGraph internamente
    logging.getLogger("langgraph_api.worker").setLevel(logging.WARNING)
    logging.getLogger("langgraph_api.worker").setLevel(logging.INFO)