import os
from datetime import datetime
import logging

# 1. Nom du fichier log
LOG_FILE = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'

# 2. Dossier "logs"
logs_dir = os.path.join(os.getcwd(), 'logs')

# 3. Créer le dossier s'il n'existe pas
os.makedirs(logs_dir, exist_ok=True)

# 4. Chemin complet du fichier log
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# 5. Configuration du logger
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
  )

# 6. Logger
logger = logging.getLogger(__name__)



