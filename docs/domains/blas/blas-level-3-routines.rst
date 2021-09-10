.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

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
         * -     :ref:`onemkl_blas_gemm`   
           -     Computes a matrix-matrix product with general matrices.   
         * -     :ref:`onemkl_blas_hemm`   
           -     Computes a matrix-matrix product where one input matrix is Hermitian and one is general.   
         * -     :ref:`onemkl_blas_herk`   
           -     Performs a Hermitian rank-k update.    
         * -     :ref:`onemkl_blas_her2k`   
           -     Performs a Hermitian rank-2k update.    
         * -     :ref:`onemkl_blas_symm`   
           -     Computes a matrix-matrix product where one input matrix is symmetric and one matrix is general.   
         * -     :ref:`onemkl_blas_syrk`   
           -     Performs a symmetric rank-k update.    
         * -     :ref:`onemkl_blas_syr2k`   
           -     Performs a symmetric rank-2k update.    
         * -     :ref:`onemkl_blas_trmm`   
           -     Computes a matrix-matrix product where one input matrix is triangular and one input matrix is general.   
         * -     :ref:`onemkl_blas_trsm`   
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
