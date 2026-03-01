pipeline {
    agent any

    environment {
        IMAGE = "2022bcs0109/wine_predict:21"
        CONTAINER = "wine-lab7-test"
        PORT = "8000"
    }

    stages {
        stage('Docker Test') {
            steps {
            sh 'docker ps'
            }
    }

        stage('Pull Image') {
            steps {
                sh "docker pull ${IMAGE}"
            }
        }
stage('Run Container') {
    steps {
        sh """
            docker stop ${CONTAINER} || true
            docker rm ${CONTAINER} || true
            docker run -d --network lab7-network \
              --name ${CONTAINER} ${IMAGE}
        """
    }
}

       stage('Wait for API') {
        steps {
            script {
                timeout(time: 60, unit: 'SECONDS') {
                    waitUntil {
                    def status = sh(
                        script: "curl -s -o /dev/null -w '%{http_code}' http://${CONTAINER}:8000/docs || true",
                        returnStdout: true
                    ).trim()

                    echo "Health check status: ${status}"
                    return (status == "200")
                 }
             }
         }
      }
   }
        stage('Valid Inference Test') {
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
                        error("Prediction field missing!")
                    }
                }
            }
        }

        stage('Invalid Inference Test') {
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

                    echo "Invalid request status: ${status}"

                    if (status == "200") {
                        error("Invalid input incorrectly returned 200!")
                    }
                }
            }
        }
        stage('Stop Container') {
            steps {
                sh """
                    docker stop ${CONTAINER} || true
                    docker rm ${CONTAINER} || true
                """
            }
        }
    }

    post {
        success {
            echo "Pipeline PASSED — Model validated successfully."
        }
        failure {
            echo "Pipeline FAILED — Model validation failed."
        }
    }
}