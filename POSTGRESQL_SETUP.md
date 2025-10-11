# PostgreSQL Setup Guide for Stock Prediction Project

This guide will help you set up PostgreSQL for the Stock Prediction project.

## Prerequisites

- Windows 10/11 (for this setup guide)
- Python 3.8+ installed
- Administrative privileges for installing PostgreSQL

## Step 1: Install PostgreSQL

### Option A: Download and Install PostgreSQL
1. Go to https://www.postgresql.org/download/windows/
2. Download the latest PostgreSQL installer for Windows
3. Run the installer as administrator
4. Follow the installation wizard:
   - Choose installation directory (default is fine)
   - Select components (keep defaults: PostgreSQL Server, pgAdmin 4, Command Line Tools)
   - Choose data directory (default is fine)
   - Set password for the `postgres` superuser account
   - Choose port (default 5432 is fine)
   - Choose locale (default is fine)

### Option B: Install via Chocolatey (if you have it)
```powershell
choco install postgresql
```

### Option C: Install via Winget
```powershell
winget install PostgreSQL.PostgreSQL
```

## Step 2: Start PostgreSQL Service

### Check if PostgreSQL is running:
```powershell
# Check if PostgreSQL service is running
Get-Service -Name "postgresql*"

# If not running, start it
Start-Service -Name "postgresql-x64-14"  # Replace with your version
```

### Alternative: Start via Services
1. Press `Win + R`, type `services.msc`
2. Find PostgreSQL service
3. Right-click and select "Start" if not running

## Step 3: Create Database

### Option A: Using psql command line
```powershell
# Connect to PostgreSQL
psql -U postgres -h localhost

# Create database
CREATE DATABASE stock_prediction_db;

# Create a user (optional but recommended)
CREATE USER stock_user WITH PASSWORD 'your_password_here';
GRANT ALL PRIVILEGES ON DATABASE stock_prediction_db TO stock_user;

# Exit psql
\q
```

### Option B: Using pgAdmin (GUI)
1. Open pgAdmin 4 (installed with PostgreSQL)
2. Connect to PostgreSQL server
3. Right-click "Databases" → "Create" → "Database"
4. Name: `stock_prediction_db`
5. Click "Save"

## Step 4: Configure Environment Variables

Create or update your `.env` file in the `backend` directory:

```env
# PostgreSQL Database Configuration
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/stock_prediction_db
DB_HOST=localhost
DB_PORT=5432
DB_NAME=stock_prediction_db
DB_USER=postgres
DB_PASSWORD=your_password

# Alternative: If you created a separate user
# DATABASE_URL=postgresql://stock_user:your_password@localhost:5432/stock_prediction_db
# DB_USER=stock_user
# DB_PASSWORD=your_password

# Database Connection Pool Configuration
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
DB_POOL_TIMEOUT=30
```

## Step 5: Test the Connection

Run the PostgreSQL test script:
```powershell
cd backend
.\venv\Scripts\python.exe test_postgresql_connection.py
```

## Step 6: Start the Application

Once PostgreSQL is set up and tested:
```powershell
cd backend
.\venv\Scripts\python.exe app.py
```

## Troubleshooting

### Connection Refused Error
- Ensure PostgreSQL service is running
- Check if port 5432 is not blocked by firewall
- Verify database credentials in `.env` file

### Authentication Failed
- Double-check username and password in `.env` file
- Ensure the user has proper permissions on the database

### Database Does Not Exist
- Create the database using the commands in Step 3
- Verify the database name in `.env` file matches the created database

### Permission Denied
- Ensure the database user has proper privileges
- Try using the `postgres` superuser for initial setup

## Docker Alternative (Optional)

If you prefer using Docker instead of installing PostgreSQL locally:

```powershell
# Run PostgreSQL in Docker
docker run --name postgres-stock \
  -e POSTGRES_PASSWORD=your_password \
  -e POSTGRES_DB=stock_prediction_db \
  -p 5432:5432 \
  -d postgres:15

# Update .env file
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/stock_prediction_db
```

## Verification

After setup, you should see:
1. PostgreSQL service running in Windows Services
2. Successful connection test from the test script
3. Database tables created automatically when the app starts
4. Stock data being stored and retrieved from PostgreSQL

## Next Steps

Once PostgreSQL is working:
1. The application will automatically create required tables
2. Stock data will be cached in PostgreSQL instead of SQLite
3. You'll have better performance and scalability
4. Data will persist across application restarts

For any issues, check the application logs or run the test script for detailed error messages.
