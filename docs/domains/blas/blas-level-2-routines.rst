.. _blas-level-2-routines:

BLAS Level 2 Routines
=====================


.. container::


   BLAS Level 2 includes routines which perform
   matrix-vector operations as described in the following table. 


   .. container:: tablenoborder


      .. list-table:: 
         :header-rows: 1

         * -     Routines
           -     Description  
         * -     :ref:`onemkl_blas_gbmv`   
           -     Matrix-vector product using a general band matrix         
         * -     :ref:`onemkl_blas_gemv`   
           -     Matrix-vector product using a general matrix     
         * -     :ref:`onemkl_blas_ger`   
           -     Rank-1 update of a general matrix     
         * -     :ref:`onemkl_blas_gerc`   
           -     Rank-1 update of a conjugated general matrix     
         * -     :ref:`onemkl_blas_geru`   
           -     Rank-1 update of a general matrix, unconjugated          
         * -     :ref:`onemkl_blas_hbmv`   
           -     Matrix-vector product using a Hermitian band matrix          
         * -     :ref:`onemkl_blas_hemv`
           -     Matrix-vector product using a Hermitian matrix          
         * -     :ref:`onemkl_blas_her`   
           -     Rank-1 update of a Hermitian matrix     
         * -     :ref:`onemkl_blas_her2`   
           -     Rank-2 update of a Hermitian matrix     
         * -     :ref:`onemkl_blas_hpmv`   
           -     Matrix-vector product using a Hermitian packed matrix          
         * -     :ref:`onemkl_blas_hpr`   
           -     Rank-1 update of a Hermitian packed matrix     
         * -     :ref:`onemkl_blas_hpr2`   
           -     Rank-2 update of a Hermitian packed matrix     
         * -     :ref:`onemkl_blas_sbmv`   
           -     Matrix-vector product using symmetric band matrix          
         * -     :ref:`onemkl_blas_spmv`   
           -     Matrix-vector product using a symmetric packed matrix          
         * -     :ref:`onemkl_blas_spr`   
           -     Rank-1 update of a symmetric packed matrix     
         * -     :ref:`onemkl_blas_spr2`   
           -     Rank-2 update of a symmetric packed matrix     
         * -     :ref:`onemkl_blas_symv`   
           -     Matrix-vector product using a symmetric matrix          
         * -     :ref:`onemkl_blas_syr`   
           -     Rank-1 update of a symmetric matrix     
         * -     :ref:`onemkl_blas_syr2`   
           -     Rank-2 update of a symmetric matrix     
         * -     :ref:`onemkl_blas_tbmv`   
           -     Matrix-vector product using a triangular band matrix          
         * -     :ref:`onemkl_blas_tbsv`   
           -     Solution of a linear system of equations with a triangular band matrix    
         * -     :ref:`onemkl_blas_tpmv`   
           -     Matrix-vector product using a triangular packed matrix          
         * -     :ref:`onemkl_blas_tpsv`   
           -     Solution of a linear system of equations with a triangular packed matrix    
         * -     :ref:`onemkl_blas_trmv`   
           -     Matrix-vector product using a triangular matrix          
         * -     :ref:`onemkl_blas_trsv`   
           -     Solution of a linear system of equations with a triangular matrix    




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

**Parent topic:** :ref:`onemkl_blas`
