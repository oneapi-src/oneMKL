.. _blas-like-extensions:

BLAS-like Extensions
====================


.. container::


   oneAPI Math Kernel Library DPC++ provides additional routines to
   extend the functionality of the BLAS routines. These include routines
   to compute many independent vector-vector and matrix-matrix operations.

   The following table lists the BLAS-like extensions with their descriptions.


   .. container:: tablenoborder


      .. list-table:: 
         :header-rows: 1

         * -     Routines
           -     Description     
         * -     :ref:`onemkl_blas_axpy_batch`   
           -     Computes groups of vector-scalar products added to a vector.
         * -     :ref:`onemkl_blas_gemm_batch`   
           -     Computes groups of matrix-matrix products with general matrices.   
         * -     :ref:`onemkl_blas_trsm_batch`   
           -     Solves a triangular matrix equation for a group of matrices.   
         * -     :ref:`onemkl_blas_gemmt`   
           -     Computes a matrix-matrix product with general matrices, but updates
                 only the upper or lower triangular part of the result matrix.
         * -     :ref:`onemkl_blas_gemm_bias`   
           -     Computes a matrix-matrix product using general integer matrices with bias
 




.. toctree::
    :hidden:

    axpy_batch
    axpby
    copy_batch
    dgmm_batch
    gemm_batch
    gemv_batch
    syrk_batch
    trsm_batch
    gemmt
    gemm_bias

**Parent topic:** :ref:`onemkl_blas`
