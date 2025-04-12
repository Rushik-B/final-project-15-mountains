# backend/worker.py
import os
import sys
from redis import Redis
from rq import Worker, Queue, Connection
import logging

# Configure logging for the worker
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure the app directory is in the Python path
# so the worker can import 'app.generate_embeddings_task' and other necessities
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# --- Preload Models/Config --- 
# IMPORTANT: Ensure global objects needed by the task (like the embedding model 
# and SessionLocal) are loaded *before* the worker starts processing jobs. 
# This avoids loading them repeatedly for each job. 
# We rely on app.py loading these globally when imported.

try:
    logger.info("Worker: Attempting to preload dependencies from app.py...")
    # Import necessary components from app that the task function needs.
    # This will trigger the global loading code in app.py
    from app import embedding_model, SessionLocal, Study, np, SQLAlchemyError
    if embedding_model:
        logger.info("Worker: Embedding model preloaded successfully.")
    else:
         # This might happen if app.py fails to load the model.
         # The task itself has checks, but good to know early.
         logger.warning("Worker: Preloading check indicates embedding_model is None.")
    if SessionLocal:
        logger.info("Worker: SessionLocal preloaded successfully.")
    else:
        logger.warning("Worker: SessionLocal preloading failed.")
    logger.info("Worker: Preloading complete.")
except ImportError as e:
     logger.error(f"Worker: Failed to import required components from app: {e}. Ensure app.py is structured correctly for imports.")
     sys.exit(1) # Cannot run tasks without app components
except Exception as e:
     logger.error(f"Worker: Error during preloading phase: {e}")
     sys.exit(1)
# --- End Preload ---


# --- Worker Setup ---
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
logger.info(f"Worker connecting to Redis at: {redis_url.split('@')[-1] if '@' in redis_url else redis_url}")

redis_conn = None
try:
    redis_conn = Redis.from_url(redis_url)
    redis_conn.ping() # Test connection
    logger.info("Worker successfully connected to Redis.")
except Exception as e:
    logger.error(f"CRITICAL: Worker failed to connect to Redis: {e}")
    sys.exit(1) # Exit if worker cannot connect to Redis

# Listen on the queue defined in app.py
listen_queues = ['embeddings']

if __name__ == '__main__':
    logger.info(f"Starting RQ worker, listening on queues: {listen_queues}")
    # Use the established connection
    with Connection(redis_conn):
        worker = Worker(map(Queue, listen_queues))
        # Set with_scheduler=True if using RQ Scheduler features
        worker.work(with_scheduler=False, logging_level='INFO') 
    logger.info("RQ worker stopped.") 