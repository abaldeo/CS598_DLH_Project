import sqlite3

def execute_sql_commands(cursor, commands):
    """Execute multiple SQL commands using the provided cursor."""
    for command in commands:
        cursor.execute(command)

def main():
    # Define the database filename
    database_filename = "Drug.db"

    # SQL commands to drop tables if they exist
    drop_table_commands = [
        "DROP TABLE IF EXISTS drug;",
        "DROP TABLE IF EXISTS event;",
        "DROP TABLE IF EXISTS event_number;",
        "DROP TABLE IF EXISTS extraction;"
    ]

    # SQL commands to create tables
    create_table_commands = [
        """CREATE TABLE drug ( 
            [index] INTEGER,
            id TEXT,
            target TEXT,
            enzyme TEXT,
            pathway TEXT,
            smile TEXT,
            name TEXT,
            category TEXT
        );""",
        """CREATE TABLE event ( 
            [index] INTEGER,
            id1 TEXT,
            name1 TEXT,
            id2 TEXT,
            name2 TEXT,
            interaction TEXT
        );""",
        """CREATE TABLE event_number ( 
            event TEXT,
            number TEXT
        );""",
        """CREATE TABLE extraction ( 
            [index] INTEGER,
            mechanism TEXT,
            action TEXT,
            drugA TEXT,
            drugB TEXT
        );"""
    ]

    # SQL commands to create indexes
    create_index_commands = [
        "CREATE INDEX ix_drug_index ON drug ([index]);",
        "CREATE INDEX ix_event_index ON event ([index]);",
        "CREATE INDEX ix_extraction_index ON extraction ([index]);"
    ]

    # Connect to the SQLite database
    with sqlite3.connect(database_filename) as conn:
        cursor = conn.cursor()

        # Execute SQL commands
        execute_sql_commands(cursor, drop_table_commands)
        execute_sql_commands(cursor, create_table_commands)
        execute_sql_commands(cursor, create_index_commands)

        # Commit the changes to the database
        conn.commit()

    print("Database and tables created successfully.")

if __name__ == "__main__":
    main()