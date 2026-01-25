"""
Database connection pool management
"""

import oracledb
from contextlib import contextmanager
from typing import Generator
from config import settings


class DatabasePool:
    """Oracle database connection pool"""

    def __init__(self):
        self._pool = None
        self._initialized = False

    def initialize(self):
        """Initialize connection pool"""
        if self._initialized:
            return

        try:
            self._pool = oracledb.create_pool(
                user=settings.oracle_username,
                password=settings.oracle_password,
                dsn=settings.oracle_dsn,
                config_dir=settings.oracle_wallet_path,
                wallet_location=settings.oracle_wallet_path,
                wallet_password=settings.oracle_password,
                min=settings.db_pool_min,
                max=settings.db_pool_max,
                increment=settings.db_pool_increment,
            )
            self._initialized = True
        except Exception as e:
            raise RuntimeError(f"Failed to initialize database pool: {e}")

    @contextmanager
    def acquire(self) -> Generator:
        """Acquire database connection from pool"""
        if not self._initialized:
            self.initialize()

        conn = self._pool.acquire()
        try:
            yield conn
        finally:
            self._pool.release(conn)

    def get_cursor(self):
        """Get a cursor from pool"""
        if not self._initialized:
            self.initialize()

        conn = self._pool.acquire()
        return conn, conn.cursor()

    def close(self):
        """Close connection pool"""
        if self._pool:
            self._pool.close()
            self._initialized = False


db_pool = DatabasePool()


def init_tables():
    """Initialize database tables"""
    with db_pool.acquire() as conn:
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT COUNT(*) FROM user_tables WHERE table_name = 'ANOMALY_SIGNALS'
            """)
            exists = cursor.fetchone()[0] > 0

            if not exists:
                cursor.execute("""
                    CREATE TABLE anomaly_signals (
                        id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                        window_start TIMESTAMP,
                        template_id VARCHAR2(100),
                        signature CLOB,
                        count NUMBER,
                        score NUMBER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()

            cursor.execute("""
                SELECT COUNT(*) FROM user_tab_columns 
                WHERE table_name = 'DOCS' AND column_name = 'TS'
            """)
            ts_exists = cursor.fetchone()[0] > 0

            if not ts_exists:
                cursor.execute("ALTER TABLE docs ADD (ts TIMESTAMP)")
                conn.commit()
        except Exception as e:
            conn.rollback()
            raise RuntimeError(f"Failed to initialize tables: {e}")
