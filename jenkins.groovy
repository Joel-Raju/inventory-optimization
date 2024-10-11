pipeline {
    agent any

    environment {
        // Set your environment variables here
        APP_NAME = "my-flask-app"
        VIRTUAL_ENV = "venv"
        PG_HOST = "your_postgres_host"
        PG_DATABASE = "your_database_name"
        PG_USER = "your_database_user"
        PG_PASSWORD = "your_database_password"
    }

    stages {
        stage('Checkout') {
            steps {
                git 'https://github.com/your-repo/your-flask-app.git'
            }
        }

        stage('Build') {
            steps {
                sh 'python -m venv $VIRTUAL_ENV'
                sh '$VIRTUAL_ENV/bin/pip install -r requirements.txt'
            }
        }

        stage('Test') {
            steps {
                sh '$VIRTUAL_ENV/bin/pytest'
            }
        }

        stage('Deploy') {
            steps {
                // Assuming you're using a WSGI server like Gunicorn
                sh '$VIRTUAL_ENV/bin/gunicorn --workers 3 --bind 0.0.0.0:8000 app:app &'
            }
        }

        stage('Database Migrations') {
            steps {
                // Assuming you're using Flask-Migrate
                sh '$VIRTUAL_ENV/bin/flask db upgrade'
            }
        }
    }

    post {
        always {
            // Clean up the workspace
            cleanWs()
        }
    }
}
