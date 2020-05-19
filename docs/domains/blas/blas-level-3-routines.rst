.. _blas-level-3-routines:

BLAS Level 3 Routines
=====================


.. container::


   BLAS Level 3 includes routines which perform
   matrix-matrix operations as described in the following table. 


   .. container:: tablenoborder


      .. list-table:: 
         :header-rows: 1

         * -     Routines
           -     Description     
         * -     \ `gemm <gemm.html>`__\   
           -     Computes a matrix-matrix product with general matrices.   
         * -     \ `hemm <hemm.html>`__\   
           -     Computes a matrix-matrix product where one input matrix is Hermitian and one is general.   
         * -     \ `herk <herk.html>`__\   
           -     Performs a Hermitian rank-k update.    
         * -     \ `her2k <her2k.html>`__\   
           -     Performs a Hermitian rank-2k update.    
         * -     \ `symm <symm.html>`__\   
           -     Computes a matrix-matrix product where one input matrix is symmetric and one matrix is general.   
         * -     \ `syrk <syrk.html>`__\   
           -     Performs a symmetric rank-k update.    
         * -     \ `syr2k <syr2k.html>`__\   
           -     Performs a symmetric rank-2k update.    
         * -     \ `trmm <trmm.html>`__\   
           -     Computes a matrix-matrix product where one input matrix is triangular and one input matrix is general.   
         * -     \ `trsm <trsm.html>`__\   
           -     Solves a triangular matrix equation (forward or backward solve).   



.. toctree::
    :hidden:

    gemm
    hemm
    herk
    her2k
    symm
    syrk
    syr2k
    trmm
    trsm

**Parent topic:** :ref:`onemkl_blas`
