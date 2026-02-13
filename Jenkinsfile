pipeline {
    agent any

    environment {
        DOCKERHUB_USER = "2022bcs0109"   // <-- your dockerhub username
        IMAGE_NAME = "wine-quality"
    }

    stages {

        stage('Setup Python Virtual Environment') {
            steps {
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Train Model') {
            steps {
                sh '''
                    . venv/bin/activate
                    python train.py
                '''
            }
        }

        stage('Read Accuracy') {
            steps {
                script {
                    def metrics = readJSON file: 'output/results/results.json'
                    env.CURRENT_ACCURACY = metrics.R2.toString()
                    echo "Current Accuracy: ${env.CURRENT_ACCURACY}"
                }
            }
        }

        stage('Compare Accuracy') {
            steps {
                script {
                    withCredentials([string(credentialsId: 'best-accuracy', variable: 'BEST_ACC')]) {

                        if (env.CURRENT_ACCURACY.toFloat() > BEST_ACC.toFloat()) {
                            env.BUILD_MODEL = "true"
                            echo "Model improved. Will build Docker image."
                        } else {
                            env.BUILD_MODEL = "false"
                            echo "Model did not improve."
                        }
                    }
                }
            }
        }

        stage('Build Docker Image') {
            when {
                expression { env.BUILD_MODEL == "true" }
            }
            steps {
                sh """
                    docker build -t ${DOCKERHUB_USER}/${IMAGE_NAME}:${BUILD_NUMBER} .
                """
            }
        }

        stage('Push Docker Image') {
            when {
                expression { env.BUILD_MODEL == "true" }
            }
            steps {
                withDockerRegistry([credentialsId: 'dockerhub-creds', url: '']) {
                    sh """
                        docker push ${DOCKERHUB_USER}/${IMAGE_NAME}:${BUILD_NUMBER}
                        docker tag ${DOCKERHUB_USER}/${IMAGE_NAME}:${BUILD_NUMBER} ${DOCKERHUB_USER}/${IMAGE_NAME}:latest
                        docker push ${DOCKERHUB_USER}/${IMAGE_NAME}:latest
                    """
                }
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'output/**', fingerprint: true
        }
    }
}
