# Setting Up Supabase PostgreSQL for Your Flask Backend

This guide walks you through setting up a Supabase PostgreSQL database for your Flask application.

## Prerequisites

1. A Supabase account (https://supabase.com)
2. Python and pip installed locally

## Step 1: Set Up the Database

1. Create a Supabase project:
   - Go to https://supabase.com and sign up
   - Create a new project
   - Set a secure database password
   - Choose a region close to your users

2. Get the database connection string:
   - Go to Settings → Database
   - Find the "Connection string" section and select "URI"
   - Copy the connection string (format: `postgresql://postgres:[YOUR-PASSWORD]@db.[YOUR-PROJECT-ID].supabase.co:5432/postgres`)
   - Replace `[YOUR-PASSWORD]` with your database password

## Step 2: Configure Environment Variables

1. Create a new `.env` file based on the `.env.example` file:
   ```
   cp .env.example .env
   ```

2. Update the `.env` file with:
   - Your Supabase PostgreSQL connection string in `DATABASE_URL`
   - Your Google API key for Gemini in `GOOGLE_API_KEY`
   - Your secret key for Flask in `SECRET_KEY`
   - Any other configuration values

## Step 3: Install Dependencies

Install all required dependencies:

```bash
pip install -r requirements.txt
```

## Step 4: Initialize and Run Migrations

If this is a fresh setup, you'll need to create and run migrations:

1. Initialize an Alembic migration:
   ```bash
   cd backend
   python -m alembic revision --autogenerate -m "initial migration"
   ```

2. Apply the migration:
   ```bash
   python -m alembic upgrade head
   ```

## Step 5: Running Your Application

Start your Flask application:

```bash
python app.py
```

## Troubleshooting

### Database Connection Issues
- Make sure `DATABASE_URL` includes `?sslmode=require` for Supabase
- Verify network access is not restricted in Supabase
- Check Supabase database logs for connection issues

### Data Migration
If you need to migrate existing data from SQLite to PostgreSQL:
1. Export data from your SQLite database using SQLite CLI or a tool like DB Browser for SQLite
2. Import the data into PostgreSQL using the Supabase dashboard or pgAdmin

## Additional Resources

- [Supabase Documentation](https://supabase.com/docs)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/) 