# This file was made by Maheen Abbasi on Oct. 4, 2026 as part of a personal project

"""
db.py: Handles SQLite persistence for saved characters in Character Studio. Characters are stored in
a local SQLite database so they survive browser refreshes and app restarts.
"""

# Imports
import os
import sqlite3
import pickle


DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "characters.db")


def get_connection():
    """
    get_connection: opens a connection to the local SQLite database.
    """
    return sqlite3.connect(DB_PATH)


def init_db():
    """
    init_db: creates the characters table if it does not already exist.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS characters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            personality TEXT NOT NULL,
            appearance TEXT NOT NULL,
            lorebook TEXT,
            avatar BLOB,
            history BLOB,
            profile BLOB,
            memory_summary TEXT,
            summarized_through INTEGER DEFAULT 0
        )
    """)

    conn.commit()
    conn.close()


def insert_character(character):
    """
    insert_character: saves a new character to the database.

    Input(s):
        character: character dictionary containing the character's details.

    Output(s):
        The database ID assigned to the new character.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO characters
        (name, personality, appearance, lorebook, avatar, history, profile, memory_summary, summarized_through)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            character["name"],
            character["personality"],
            character["appearance"],
            character.get("lorebook", ""),
            pickle.dumps(character.get("avatar")),
            pickle.dumps(character.get("history", [])),
            pickle.dumps(character.get("profile", {})),
            character.get("memory_summary", ""),
            character.get("summarized_through", 0),
        ),
    )

    db_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return db_id


def update_character(db_id, character):
    """
    update_character: updates an existing saved character in the database.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE characters
        SET name = ?,
            personality = ?,
            appearance = ?,
            lorebook = ?,
            avatar = ?,
            history = ?,
            profile = ?,
            memory_summary = ?,
            summarized_through = ?
        WHERE id = ?
        """,
        (
            character["name"],
            character["personality"],
            character["appearance"],
            character.get("lorebook", ""),
            pickle.dumps(character.get("avatar")),
            pickle.dumps(character.get("history", [])),
            pickle.dumps(character.get("profile", {})),
            character.get("memory_summary", ""),
            character.get("summarized_through", 0),
            db_id,
        ),
    )

    conn.commit()
    conn.close()


def load_all_characters():
    """
    load_all_characters: loads every saved character from SQLite and converts
    each database row back into the dictionary format used by app.py.

    Output(s):
        A list of character dictionaries ordered from oldest to newest.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            name,
            personality,
            appearance,
            lorebook,
            avatar,
            history,
            profile,
            memory_summary,
            summarized_through
        FROM characters
        ORDER BY id ASC
    """)

    rows = cursor.fetchall()
    conn.close()

    characters = []

    for row in rows:
        (
            db_id,
            name,
            personality,
            appearance,
            lorebook,
            avatar,
            history,
            profile,
            memory_summary,
            summarized_through,
        ) = row

        characters.append({
            "db_id": db_id,
            "name": name,
            "personality": personality,
            "appearance": appearance,
            "lorebook": lorebook or "",
            "avatar": pickle.loads(avatar) if avatar else None,
            "history": pickle.loads(history) if history else [],
            "profile": pickle.loads(profile) if profile else {},
            "memory_summary": memory_summary or "",
            "summarized_through": summarized_through or 0,
        })

    return characters


def delete_character(db_id):
    """
    delete_character: removes a character from the database by its ID.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM characters WHERE id = ?",
        (db_id,),
    )

    conn.commit()
    conn.close()


def delete_all_characters():
    """
    delete_all_characters: removes every character from the database.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM characters")
    cursor.execute("DELETE FROM sqlite_sequence WHERE name = 'characters'")

    conn.commit()
    conn.close()