#!/bin/bash
set -e

# Docker entrypoint script for Python UV Template

echo "🚀 Starting Python UV Template..."

# Function to wait for database
wait_for_db() {
    if [ -n "$DATABASE_URL" ]; then
        echo "⏳ Waiting for database..."
        while ! python -c "
import sys
try:
    import psycopg2
    import os
    db_url = os.environ.get('DATABASE_URL')
    if db_url:
        conn = psycopg2.connect(db_url)
        conn.close()
        print('✅ Database is ready!')
except Exception as e:
    print(f'❌ Database not ready: {e}')
    sys.exit(1)
        "; do
            echo "Database is unavailable - sleeping"
            sleep 1
        done
    fi
}

# Function to run database migrations (if applicable)
run_migrations() {
    if [ "$RUN_MIGRATIONS" = "true" ]; then
        echo "🔄 Running database migrations..."
        # Add your migration commands here
        # python -m alembic upgrade head
        echo "✅ Migrations completed"
    fi
}

# Function to collect static files (if applicable)
collect_static() {
    if [ "$COLLECT_STATIC" = "true" ]; then
        echo "📦 Collecting static files..."
        # Add your static collection commands here
        echo "✅ Static files collected"
    fi
}

# Main execution
main() {
    echo "Environment: ${ENVIRONMENT:-production}"
    echo "Python path: $PYTHONPATH"

    # Wait for external services
    wait_for_db

    # Run setup tasks
    run_migrations
    collect_static

    echo "🎯 Starting application..."

    # Execute the main command
    exec "$@"
}

# Run main function with all arguments
main "$@"
