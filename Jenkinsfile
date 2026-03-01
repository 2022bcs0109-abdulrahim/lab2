pipeline {
    agent any

    environment {
        DOCKERHUB_USER = "2022bcs0109"
        IMAGE_NAME = "wine-quality"
        IMAGE_TAG = "latest"
        CONTAINER_NAME = "wine-validation-container"
        PORT = "8000"
    }

    stages {

        stage('Pull Image') {
            steps {
                withDockerRegistry([credentialsId: 'dockerhub-creds', url: '']) {
                    sh """
                        docker pull ${DOCKERHUB_USER}/${IMAGE_NAME}:${IMAGE_TAG}
                    """
                }
            }
        }

        stage('Run Container') {
            steps {
                sh """
                    docker run -d -p ${PORT}:8000 --name ${CONTAINER_NAME} \
                    ${DOCKERHUB_USER}/${IMAGE_NAME}:${IMAGE_TAG}
                """
            }
        }

        stage('Wait for Service Readiness') {
            steps {
                script {
                    timeout(time: 60, unit: 'SECONDS') {
                        waitUntil {
                            def status = sh(
                                script: "curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/docs || true",
                                returnStdout: true
                            ).trim()

                            echo "Health check status: ${status}"
                            return (status == "200")
                        }
                    }
                }
            }
        }

        stage('Send Valid Inference Request') {
            steps {
                script {
                    def response = sh(
                        script: """
                            curl -s -X POST http://localhost:${PORT}/predict \
                            -H "Content-Type: application/json" \
                            -d @tests/valid.json
                        """,
                        returnStdout: true
                    ).trim()

                    echo "Valid Response: ${response}"

                    if (!response.contains("prediction")) {
                        error("Prediction field missing in valid response!")
                    }
                }
            }
        }

        stage('Send Invalid Inference Request') {
            steps {
                script {
                    def status = sh(
                        script: """
                            curl -s -o /dev/null -w '%{http_code}' \
                            -X POST http://localhost:${PORT}/predict \
                            -H "Content-Type: application/json" \
                            -d @tests/invalid.json
                        """,
                        returnStdout: true
                    ).trim()

                    echo "Invalid request status code: ${status}"

                    if (status == "200") {
                        error("Invalid input incorrectly returned 200!")
                    }
                }
            }
        }

        stage('Stop Container') {
            steps {
                sh """
                    docker stop ${CONTAINER_NAME} || true
                    docker rm ${CONTAINER_NAME} || true
                """
            }
        }
    }

    post {
        failure {
            echo "Pipeline FAILED — Model validation failed."
        }
        success {
            echo "Pipeline PASSED — Model inference validated successfully."
        }
    }
}