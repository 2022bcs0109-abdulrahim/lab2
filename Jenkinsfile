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
                    curl -s "http://${CONTAINER}:8000/predict?\
fixed_acidity=7.4&\
volatile_acidity=0.7&\
citric_acid=0.0&\
residual_sugar=1.9&\
chlorides=0.076&\
free_sulfur_dioxide=11.0&\
total_sulfur_dioxide=34.0&\
density=0.9978&\
ph=3.51&\
sulphates=0.56&\
alcohol=9.4"
                """,
                returnStdout: true
            ).trim()

            echo "Valid Response: ${response}"

            if (!response.contains("predicted_quality")) {
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
                    curl -s -o /dev/null -w "%{http_code}" \
                    "http://${CONTAINER}:8000/predict"
                """,
                returnStdout: true
            ).trim()

            echo "Invalid request status: ${status}"

            if (status != "422") {
                error("Expected 422 for invalid input!")
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