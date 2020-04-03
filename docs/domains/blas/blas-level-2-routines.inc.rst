.. _blas-level-2-routines:

BLAS Level 2 Routines
=====================


.. container::


   This section describes BLAS Level 2 routines, which perform
   matrix-vector operations. The following table lists the BLAS Level 2
   routine groups and the data types associated with them.


   .. container:: tablenoborder


      .. list-table:: 
         :header-rows: 1

         * -     Routine or Function Group with SYCL Buffer
           -     Data Types     
           -     Description     
         * -           \ `gbmv <gbmv.html>`__\    
           -     float, double, std::complex<float>,       std::complex<double>    
           -     Matrix-vector product using a general band matrix          
         * -           \ `gemv <gemv.html>`__\    
           -     float, double, std::complex<float>,       std::complex<double>    
           -     Matrix-vector product using a general matrix     
         * -           \ `ger <ger.html>`__\    
           -     float, double     
           -     Rank-1 update of a general matrix     
         * -           \ `gerc <gerc.html>`__\    
           -     std::complex<float>, std::complex<double>     
           -     Rank-1 update of a conjugated general matrix     
         * -           \ `geru <geru.html>`__\    
           -     std::complex<float>, std::complex<double>     
           -     Rank-1 update of a general matrix, unconjugated          
         * -           \ `hbmv <hbmv.html>`__\    
           -     std::complex<float>, std::complex<double>     
           -     Matrix-vector product using a Hermitian band matrix          
         * -           \ `hemv <hemv.html>`__\    
           -     std::complex<float>, std::complex<double>     
           -     Matrix-vector product using a Hermitian matrix          
         * -           \ `her <her.html>`__\    
           -     std::complex<float>, std::complex<double>     
           -     Rank-1 update of a Hermitian matrix     
         * -           \ `her2 <her2.html>`__\    
           -     std::complex<float>, std::complex<double>     
           -     Rank-2 update of a Hermitian matrix     
         * -           \ `hpmv <hpmv.html>`__\    
           -     std::complex<float>, std::complex<double>     
           -     Matrix-vector product using a Hermitian packed matrix          
         * -           \ `hpr <hpr.html>`__\    
           -     std::complex<float>, std::complex<double>     
           -     Rank-1 update of a Hermitian packed matrix     
         * -           \ `hpr2 <hpr2.html>`__\    
           -     std::complex<float>, std::complex<double>     
           -     Rank-2 update of a Hermitian packed matrix     
         * -           \ `sbmv <sbmv.html>`__\    
           -     float, double     
           -     Matrix-vector product using symmetric band matrix          
         * -           \ `spmv <spmv.html>`__\    
           -     float, double     
           -     Matrix-vector product using a symmetric packed matrix          
         * -           \ `spr <spr.html>`__\    
           -     float, double     
           -     Rank-1 update of a symmetric packed matrix     
         * -           \ `spr2 <spr2.html>`__\    
           -     float, double     
           -     Rank-2 update of a symmetric packed matrix     
         * -           \ `symv <symv.html>`__\    
           -     float, double     
           -     Matrix-vector product using a symmetric matrix          
         * -           \ `syr <syr.html>`__\    
           -     float, double     
           -     Rank-1 update of a symmetric matrix     
         * -           \ `syr2 <syr2.html>`__\    
           -     float, double     
           -     Rank-2 update of a symmetric matrix     
         * -           \ `tbmv <tbmv.html>`__\    
           -     float, double, std::complex<float>,       std::complex<double>    
           -     Matrix-vector product using a triangular band matrix          
         * -           \ `tbsv <tbsv.html>`__\    
           -     float, double, std::complex<float>,       std::complex<double>    
           -     Solution of a linear system of equations with a       triangular band matrix    
         * -           \ `tpmv <tpmv.html>`__\    
           -     float, double, std::complex<float>,       std::complex<double>    
           -     Matrix-vector product using a triangular packed matrix          
         * -           \ `tpsv <tpsv.html>`__\    
           -     float, double, std::complex<float>,       std::complex<double>    
           -     Solution of a linear system of equations with a       triangular packed matrix    
         * -           \ `trmv <trmv.html>`__\    
           -     float, double, std::complex<float>,       std::complex<double>    
           -     Matrix-vector product using a triangular matrix          
         * -           \ `trsv <trsv.html>`__\    
           -     float, double, std::complex<float>,       std::complex<double>    
           -     Solution of a linear system of equations with a       triangular matrix    




.. toctree::
    :hidden:

    gbmv
    gemv
    ger
    gerc
    geru
    hbmv
    hemv
    her
    her2
    hpmv
    hpr
    hpr2
    sbmv
    spmv
    spr
    spr2
    symv
    syr
    syr2
    tbmv
    tbsv
    tpmv
    tpsv
    trmv
    trsv
