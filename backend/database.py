"""
SteppeDNA — SQLite Database Persistence

Stores analysis history server-side for cross-session tracking and analytics.
Uses synchronous sqlite3 wrapped in FastAPI async endpoints.
"""
import os
import sqlite3
import json
import time
import logging

logger = logging.getLogger("steppedna.db")

DB_PATH = os.getenv("STEPPEDNA_DB_PATH", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "steppedna.db"
))

def _get_conn():
    """Get a thread-local SQLite connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent read performance
    conn.execute("PRAGMA busy_timeout=5000")
    return conn

def init_db():
    """Initialize database tables if they don't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = _get_conn()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                gene TEXT NOT NULL,
                cdna_pos INTEGER,
                aa_ref TEXT,
                aa_alt TEXT,
                aa_pos INTEGER,
                mutation TEXT,
                prediction TEXT,
                probability REAL,
                risk_tier TEXT,
                confidence_label TEXT,
                ci_lower REAL,
                ci_upper REAL,
                ip_hash TEXT,
                request_id TEXT,
                latency_ms REAL,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_analyses_gene ON analyses(gene);
            CREATE INDEX IF NOT EXISTS idx_analyses_timestamp ON analyses(timestamp);
            CREATE INDEX IF NOT EXISTS idx_analyses_prediction ON analyses(prediction);

            CREATE TABLE IF NOT EXISTS vcf_uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                filename TEXT,
                gene TEXT NOT NULL,
                total_variants INTEGER,
                missense_found INTEGER,
                pathogenic_count INTEGER,
                benign_count INTEGER,
                ip_hash TEXT,
                latency_ms REAL,
                created_at TEXT DEFAULT (datetime('now'))
            );
        """)
        conn.commit()
        logger.info(f"[DB] Initialized at {DB_PATH}")
    finally:
        conn.close()

def record_analysis(gene, cdna_pos, aa_ref, aa_alt, aa_pos, mutation,
                     prediction, probability, risk_tier, confidence_label,
                     ci_lower, ci_upper, ip_hash=None, request_id=None, latency_ms=None):
    """Record a single variant analysis."""
    conn = _get_conn()
    try:
        conn.execute(
            """INSERT INTO analyses
               (timestamp, gene, cdna_pos, aa_ref, aa_alt, aa_pos, mutation,
                prediction, probability, risk_tier, confidence_label,
                ci_lower, ci_upper, ip_hash, request_id, latency_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (time.time(), gene, cdna_pos, aa_ref, aa_alt, aa_pos, mutation,
             prediction, probability, risk_tier, confidence_label,
             ci_lower, ci_upper, ip_hash, request_id, latency_ms)
        )
        conn.commit()
    except Exception as e:
        logger.warning(f"[DB] Failed to record analysis: {e}")
    finally:
        conn.close()

def record_vcf_upload(filename, gene, total_variants, missense_found,
                       pathogenic_count, benign_count, ip_hash=None, latency_ms=None):
    """Record a VCF upload analysis."""
    conn = _get_conn()
    try:
        conn.execute(
            """INSERT INTO vcf_uploads
               (timestamp, filename, gene, total_variants, missense_found,
                pathogenic_count, benign_count, ip_hash, latency_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (time.time(), filename, gene, total_variants, missense_found,
             pathogenic_count, benign_count, ip_hash, latency_ms)
        )
        conn.commit()
    except Exception as e:
        logger.warning(f"[DB] Failed to record VCF upload: {e}")
    finally:
        conn.close()

def get_recent_analyses(limit=50):
    """Get recent analyses for the history API."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            """SELECT gene, cdna_pos, aa_ref, aa_alt, aa_pos, prediction,
                      probability, risk_tier, confidence_label, timestamp, latency_ms
               FROM analyses ORDER BY timestamp DESC LIMIT ?""",
            (limit,)
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

def get_analysis_stats():
    """Get aggregate analysis statistics."""
    conn = _get_conn()
    try:
        total = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
        by_gene = dict(conn.execute(
            "SELECT gene, COUNT(*) FROM analyses GROUP BY gene"
        ).fetchall())
        by_pred = dict(conn.execute(
            "SELECT prediction, COUNT(*) FROM analyses GROUP BY prediction"
        ).fetchall())
        vcf_total = conn.execute("SELECT COUNT(*) FROM vcf_uploads").fetchone()[0]

        # Average latency
        avg_latency = conn.execute(
            "SELECT AVG(latency_ms) FROM analyses WHERE latency_ms IS NOT NULL"
        ).fetchone()[0]

        return {
            "total_analyses": total,
            "total_vcf_uploads": vcf_total,
            "by_gene": by_gene,
            "by_prediction": by_pred,
            "avg_latency_ms": round(avg_latency or 0, 1),
        }
    finally:
        conn.close()
