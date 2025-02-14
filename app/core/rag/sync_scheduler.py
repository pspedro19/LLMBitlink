# core/rag/sync_scheduler.py

import asyncio
import logging
from datetime import datetime
from typing import Optional
from core.rag.config import Config
from core.rag.sync_manager import SyncManager

logger = logging.getLogger(__name__)

class SyncScheduler:
    def __init__(self, sync_manager: SyncManager, interval_seconds: int = 300):
        self.sync_manager = sync_manager
        self.interval = interval_seconds
        self.is_running = False
        self.last_sync: Optional[datetime] = None
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Inicia el proceso de sincronización automática."""
        if self.is_running:
            logger.warning("Sync scheduler is already running")
            return

        self.is_running = True
        self._task = asyncio.create_task(self._sync_loop())
        logger.info("Sync scheduler started")

    async def stop(self):
        """Detiene el proceso de sincronización automática."""
        if not self.is_running:
            return

        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Sync scheduler stopped")

    async def _sync_loop(self):
        """Loop principal de sincronización."""
        while self.is_running:
            try:
                processed, failed = await self.sync_manager.synchronize()
                self.last_sync = datetime.now()
                
                logger.info(
                    f"Sync completed - Processed: {processed}, Failed: {failed}, "
                    f"Time: {self.last_sync.isoformat()}"
                )

                # Verificar estado de sincronización
                status = self.sync_manager.verify_sync_status()
                if not status.get('is_synced', False):
                    logger.warning("Sync verification failed: PostgreSQL and FAISS are not in sync")

            except Exception as e:
                logger.error(f"Error in sync loop: {e}")

            await asyncio.sleep(self.interval)

    def get_status(self) -> dict:
        """Retorna el estado actual del scheduler."""
        return {
            'is_running': self.is_running,
            'last_sync': self.last_sync.isoformat() if self.last_sync else None,
            'interval_seconds': self.interval,
            'next_sync': (
                (self.last_sync + asyncio.timedelta(seconds=self.interval)).isoformat()
                if self.last_sync else None
            )
        }