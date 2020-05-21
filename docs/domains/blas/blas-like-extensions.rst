.. _blas-like-extensions:

BLAS-like Extensions
====================


.. container::


   oneAPI Math Kernel Library DPC++ provides additional routines to
   extend the functionality of the BLAS routines. These include routines
   to compute many independent matrix-matrix products.

   The following table lists the BLAS-like Extensions with their descriptions.


   .. container:: tablenoborder


      .. list-table:: 
         :header-rows: 1

         * -     Routines
           -     Description     
         * -     \ `gemm_batch <gemm_batch.html>`__\   
           -     Computes groups of matrix-matrix product with general       matrices.   
         * -     \ `trsm_batch <trsm_batch.html>`__\   
           -     Solves a triangular matrix equation for a group of       matrices.   
         * -     \ `gemmt <gemmt.html>`__\   
           -     Computes a matrix-matrix product with general matrices, but updates
                 only the upper or lower triangular part of the result matrix.
         * -     \ `gemm_ext <gemm_ext.html>`__\   
           -     Computes a matrix-matrix product with general matrices
 




.. toctree::
    :hidden:

    axpy_batch
    gemm_batch
    trsm_batch
    gemmt
    gemm_ext

**Parent topic:** :ref:`onemkl_blas`
