.. _blas-level-3-routines:

BLAS Level 3 Routines
=====================


.. container::


   BLAS Level 3 routines perform matrix-matrix operations. The following
   table lists the BLAS Level 3 routine groups and the data types
   associated with them.


   .. container:: tablenoborder


      .. list-table:: 
         :header-rows: 1

         * -     Routine or Function Group with SYCL Buffer
           -     Data Types     
           -     Description     
         * -           \ `gemm <gemm.html>`__\    
           -     float, double, std::complex<float>,       std::complex<double>    
           -     Computes a matrix-matrix product with general       matrices.   
         * -           \ `hemm <hemm.html>`__\    
           -     std::complex<float>, std::complex<double>     
           -     Computes a matrix-matrix product where one input       matrix is Hermitian and one is general.   
         * -           \ `herk <herk.html>`__\    
           -     std::complex<float>, std::complex<double>     
           -     Performs a Hermitian rank-k update.    
         * -           \ `her2k <her2k.html>`__\    
           -     std::complex<float>, std::complex<double>     
           -     Performs a Hermitian rank-2k update.    
         * -           \ `symm <symm.html>`__\    
           -     float, double, std::complex<float>,       std::complex<double>    
           -     Computes a matrix-matrix product where one input       matrix is symmetric and one matrix is general.   
         * -           \ `syrk <syrk.html>`__\    
           -     float, double, std::complex<float>,       std::complex<double>    
           -     Performs a symmetric rank-k update.    
         * -           \ `syr2k <syr2k.html>`__\    
           -     float, double, std::complex<float>,       std::complex<double>    
           -     Performs a symmetric rank-2k update.    
         * -           \ `trmm <trmm.html>`__\    
           -     float, double, std::complex<float>,       std::complex<double>    
           -     Computes a matrix-matrix product where one input       matrix is triangular and one input matrix is general.   
         * -           \ `trsm <trsm.html>`__\    
           -     float, double, std::complex<float>,       std::complex<double>    
           -     Solves a triangular matrix equation (forward or       backward solve).   




   -  


      .. container::
         :name: LI_21BA86AC0A4942A79BA0C7DC4ABC50C4


         The BLAS functions are blocked where possible to restructure
         the code in a way that increases the localization of data
         reference, enhances cache memory use, and reduces the
         dependency on the memory bus.


   -  


      .. container::
         :name: LI_9D82DEDFA672416D9B3EA8C9C2B6F0A3


         The code is distributed across the processors to maximize
         parallelism.


.. toctree::
    :hidden:

    gemm
    hemm
    her2k
    herk
    symm
    syr2k
    syrk
    trmm
    trsm
