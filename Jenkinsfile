pipeline {
    agent any

    environment {
        DOCKER_IMAGE = "wine-quality"
    }

    stages {

        // ❌ DO NOT ADD MANUAL CHECKOUT
        // Jenkins already checks out the repo automatically

        stage('Setup Python Virtual Environment') {
            steps {
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
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
                    def metrics = readJSON file: 'output/results/metrics.json'
                    env.CURRENT_ACCURACY = metrics.r2.toString()
                    echo "Current Accuracy: ${env.CURRENT_ACCURACY}"
                }
            }
        }

        stage('Compare Accuracy') {
            steps {
                script {
                    withCredentials([string(credentialsId: 'best-accuracy', variable: 'BEST_ACC')]) {

                        echo "Best Accuracy: ${BEST_ACC}"

                        if (env.CURRENT_ACCURACY.toFloat() > BEST_ACC.toFloat()) {
                            env.BUILD_MODEL = "true"
                            echo "Model improved!"
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
                script {
                    docker.build("${DOCKER_IMAGE}:${BUILD_NUMBER}")
                }
            }
        }

        stage('Push Docker Image') {
            when {
                expression { env.BUILD_MODEL == "true" }
            }
            steps {
                script {
                    docker.withRegistry('', 'dockerhub-creds') {
                        docker.image("${DOCKER_IMAGE}:${BUILD_NUMBER}").push()
                        docker.image("${DOCKER_IMAGE}:${BUILD_NUMBER}").push("latest")
                    }
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
