"""
Scheduled Ingestion Runner
Keeps the vector store fresh with daily rating action updates.
"""

import logging
from apscheduler.schedulers.blocking import BlockingScheduler
from ingestion.pipeline import ingest_all_agencies

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

scheduler = BlockingScheduler()

@scheduler.scheduled_job("cron", hour=6, minute=0)  # 6am daily
def daily_ingestion():
    logger.info("Starting scheduled ingestion...")
    results = ingest_all_agencies()
    logger.info(f"Ingestion complete: {results}")

if __name__ == "__main__":
    logger.info("Scheduler started — will run ingestion daily at 6am")
    scheduler.start()
