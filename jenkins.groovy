pipeline {
    agent any

    environment {
        APP_NAME = "inventory-optimization"
        VIRTUAL_ENV = "venv"
        PG_HOST = "localhost" 
        PG_DATABASE = "inventory_optimization"
        PG_USER = "postgres"
        PG_PASSWORD = "postgres"
    }

    stages {
        stage('Checkout') {
            steps {
                git 'https://github.com/Joel-Raju/inventory-optimization.git'
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
                
                sh '$VIRTUAL_ENV/bin/gunicorn --workers 3 --bind 0.0.0.0:8000 app:app &'
            }
        }

        stage('Database Migrations') {
            steps {
                sh '$VIRTUAL_ENV/bin/flask db upgrade'
            }
        }
    }

    post {
        always {
            cleanWs()
        }
    }
}