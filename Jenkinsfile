pipeline {
    agent {
        dockerfile {
            args "-u root --entrypoint=''"
        }
    }
    stages {
        stage("Build project"){
            steps{
                sh '''
                . /opt/intel/oneapi/setvars.sh
                dpcpp --version
                conan --version
                cd /home/oneMKL/BUILD
                conan install .. -pr inteldpcpp_lnx --build missing
                '''
                // NETLIB Package for LAPACK is no longer available on the official Conan repo.
                // TODO: Fix Reference BLAS and LAPACK in Conan builds.
            }
        }
    }
}