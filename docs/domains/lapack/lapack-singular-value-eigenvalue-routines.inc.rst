.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemkl_lapack-singular-value-eigenvalue-routines:

LAPACK Singular Value and Eigenvalue Problem Routines
=====================================================


.. container::


   LAPACK Singular Value and Eigenvalue Problem routines are used for
   singular value and eigenvalue problems, and for performing a number of related
   computational tasks. The following table lists the LAPACK Singular Value and 
   Eigenvalue Problem routine groups.


   .. container:: tablenoborder


      .. list-table:: 
         :header-rows: 1

         * -     Routines
           -     Scratchpad Size Routines
           -     Description     
         * -     :ref:`onemkl_lapack_gebrd`
           -     :ref:`onemkl_lapack_gebrd_scratchpad_size`
           -     Reduces a general matrix to bidiagonal form.   
         * -     :ref:`onemkl_lapack_gesvd`
           -     :ref:`onemkl_lapack_gesvd_scratchpad_size`
           -     Computes the singular value decomposition of a general rectangular matrix.
         * -     :ref:`onemkl_lapack_heevd`
           -     :ref:`onemkl_lapack_heevd_scratchpad_size`
           -     Computes all eigenvalues and, optionally, all eigenvectors of a complex Hermitian matrix using divide and conquer algorithm.
         * -     :ref:`onemkl_lapack_hegvd`
           -     :ref:`onemkl_lapack_hegvd_scratchpad_size`
           -     Computes all eigenvalues and, optionally, all eigenvectors of a complex generalized Hermitian definite eigenproblem using divide and conquer algorithm.
         * -     :ref:`onemkl_lapack_hetrd`
           -     :ref:`onemkl_lapack_hetrd_scratchpad_size`
           -     Reduces a complex Hermitian matrix to tridiagonal form.
         * -     :ref:`onemkl_lapack_orgbr`
           -     :ref:`onemkl_lapack_orgbr_scratchpad_size`
           -     Generates the real orthogonal matrix :math:`Q` or :math:`P^T` determined by gebrd.
         * -     :ref:`onemkl_lapack_orgtr`
           -     :ref:`onemkl_lapack_orgtr_scratchpad_size`
           -     Generates the real orthogonal matrix :math:`Q` determined by sytrd.
         * -     :ref:`onemkl_lapack_ormtr`
           -     :ref:`onemkl_lapack_ormtr_scratchpad_size`
           -     Multiplies a real matrix by the orthogonal matrix :math:`Q` determined by sytrd.
         * -     :ref:`onemkl_lapack_syevd`
           -     :ref:`onemkl_lapack_syevd_scratchpad_size`
           -     Computes all eigenvalues and, optionally, all eigenvectors of a real symmetric matrix using divide and conquer algorithm.
         * -     :ref:`onemkl_lapack_sygvd`
           -     :ref:`onemkl_lapack_sygvd_scratchpad_size`
           -     Computes all eigenvalues and, optionally, all eigenvectors of a real generalized symmetric definite eigenproblem using divide and conquer algorithm.
         * -     :ref:`onemkl_lapack_sytrd`
           -     :ref:`onemkl_lapack_sytrd_scratchpad_size`
           -     Reduces a real symmetric matrix to tridiagonal form.
         * -     :ref:`onemkl_lapack_ungbr`
           -     :ref:`onemkl_lapack_ungbr_scratchpad_size`
           -     Generates the complex unitary matrix :math:`Q` or :math:`P^T` determined by gebrd.
         * -     :ref:`onemkl_lapack_ungtr`
           -     :ref:`onemkl_lapack_ungtr_scratchpad_size`
           -     Generates the complex unitary matrix :math:`Q` determined by hetrd.
         * -     :ref:`onemkl_lapack_unmtr`
           -     :ref:`onemkl_lapack_unmtr_scratchpad_size`
           -     Multiplies a complex matrix by the unitary matrix :math:`Q` determined by hetrd.




.. toctree::
    :hidden:

    gebrd
    gebrd_scratchpad_size
    gesvd
    gesvd_scratchpad_size
    heevd
    heevd_scratchpad_size
    hegvd
    hegvd_scratchpad_size
    hetrd
    hetrd_scratchpad_size
    orgbr
    orgbr_scratchpad_size
    orgtr
    orgtr_scratchpad_size
    ormtr
    ormtr_scratchpad_size
    syevd
    syevd_scratchpad_size
    sygvd
    sygvd_scratchpad_size
    sytrd
    sytrd_scratchpad_size
    ungbr
    ungbr_scratchpad_size
    ungtr
    ungtr_scratchpad_size
    unmtr
    unmtr_scratchpad_size
