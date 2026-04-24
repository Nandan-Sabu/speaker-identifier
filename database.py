import os
import psycopg2
import numpy as np
import io

DATABASE_URL = os.environ.get("DATABASE_URL")


def get_connection():
    return psycopg2.connect(DATABASE_URL)


def setup_database():
    """Create the speakers table if it doesn't exist."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS speakers (
            id SERIAL PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            embedding BYTEA NOT NULL
        );
    """)
    conn.commit()
    cur.close()
    conn.close()


def save_profile(name: str, embedding: np.ndarray):
    """Save or overwrite a speaker profile in the database."""
    # Serialize numpy array to bytes
    buf = io.BytesIO()
    np.save(buf, embedding)
    embedding_bytes = buf.getvalue()

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO speakers (name, embedding)
        VALUES (%s, %s)
        ON CONFLICT (name) DO UPDATE SET embedding = EXCLUDED.embedding;
    """, (name, psycopg2.Binary(embedding_bytes)))
    conn.commit()
    cur.close()
    conn.close()


def load_profiles() -> dict:
    """Load all speaker profiles from the database."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT name, embedding FROM speakers;")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    profiles = {}
    for name, embedding_bytes in rows:
        buf = io.BytesIO(bytes(embedding_bytes))
        profiles[name] = np.load(buf)

    return profiles


def delete_profile(name: str):
    """Delete a speaker profile from the database."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM speakers WHERE name = %s;", (name,))
    conn.commit()
    cur.close()
    conn.close()


def list_profiles() -> list:
    """Return a list of all saved speaker names."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT name FROM speakers;")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [row[0] for row in rows]