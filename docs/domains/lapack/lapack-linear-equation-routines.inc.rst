.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemkl_lapack-linear-equation-routines:

LAPACK Linear Equation Routines
===============================


.. container::


   LAPACK Linear Equation routines are used for factoring a matrix,
   solving a system of linear equations, solving linear least squares problems,
   and inverting a matrix. The following table lists the LAPACK Linear Equation
   routine groups.


   .. container:: tablenoborder


      .. list-table:: 
         :header-rows: 1

         * -     Routines
           -     Scratchpad Size Routines
           -     Description     
         * -     :ref:`onemkl_lapack_geqrf`
           -     :ref:`onemkl_lapack_geqrf_scratchpad_size`
           -     Computes the QR factorization of a general m-by-n matrix.
         * -     :ref:`onemkl_lapack_gerqf`
           -     :ref:`onemkl_lapack_gerqf_scratchpad_size`
           -     Computes the RQ factorization of a general m-by-n matrix.
         * -     :ref:`onemkl_lapack_getrf`
           -     :ref:`onemkl_lapack_getrf_scratchpad_size`
           -     Computes the LU factorization of a general m-by-n matrix.   
         * -     :ref:`onemkl_lapack_getri`
           -     :ref:`onemkl_lapack_getri_scratchpad_size`
           -     Computes the inverse of an LU-factored general matrix.   
         * -     :ref:`onemkl_lapack_getrs`
           -     :ref:`onemkl_lapack_getrs_scratchpad_size`
           -     Solves a system of linear equations with an LU-factored square coefficient matrix, with multiple right-hand sides.    
         * -     :ref:`onemkl_lapack_hetrf`
           -     :ref:`onemkl_lapack_hetrf_scratchpad_size`
           -     Computes the Bunch-Kaufman factorization of a complex Hermitian matrix.
         * -     :ref:`onemkl_lapack_orgqr`
           -     :ref:`onemkl_lapack_orgqr_scratchpad_size`
           -     Generates the real orthogonal matrix :math:`Q` of the QR factorization formed by geqrf.
         * -     :ref:`onemkl_lapack_ormqr`
           -     :ref:`onemkl_lapack_ormqr_scratchpad_size`
           -     Multiplies a real matrix by the orthogonal matrix :math:`Q` of the QR factorization formed by geqrf.
         * -     :ref:`onemkl_lapack_ormrq`
           -     :ref:`onemkl_lapack_ormrq_scratchpad_size`
           -     Multiplies a real matrix by the orthogonal matrix :math:`Q` of the RQ factorization formed by gerqf.
         * -     :ref:`onemkl_lapack_potrf`
           -     :ref:`onemkl_lapack_potrf_scratchpad_size`
           -     Computes the Cholesky factorization of a symmetric (Hermitian) positive-definite matrix.   
         * -     :ref:`onemkl_lapack_potri`
           -     :ref:`onemkl_lapack_potri_scratchpad_size`
           -     Computes the inverse of a Cholesky-factored symmetric (Hermitian) positive-definite matrix.   
         * -     :ref:`onemkl_lapack_potrs`
           -     :ref:`onemkl_lapack_potrs_scratchpad_size`
           -     Solves a system of linear equations with a Cholesky-factored symmetric (Hermitian) positive-definite coefficient matrix, with multiple right-hand sides.    
         * -     :ref:`onemkl_lapack_sytrf`
           -     :ref:`onemkl_lapack_sytrf_scratchpad_size`
           -     Computes the Bunch-Kaufman factorization of a symmetric matrix.   
         * -     :ref:`onemkl_lapack_trtrs`
           -     :ref:`onemkl_lapack_trtrs_scratchpad_size`
           -     Solves a system of linear equations with a triangular coefficient matrix, with multiple right-hand sides.    
         * -     :ref:`onemkl_lapack_ungqr`
           -     :ref:`onemkl_lapack_ungqr_scratchpad_size`
           -     Generates the complex unitary matrix :math:`Q` of the QR factorization formed by geqrf.
         * -     :ref:`onemkl_lapack_unmqr`
           -     :ref:`onemkl_lapack_unmqr_scratchpad_size`
           -     Multiplies a complex matrix by the unitary matrix :math:`Q` of the QR factorization formed by geqrf.
         * -     :ref:`onemkl_lapack_unmrq`
           -     :ref:`onemkl_lapack_unmrq_scratchpad_size`
           -     Multiplies a complex matrix by the unitary matrix :math:`Q` of the RQ factorization formed by gerqf.





.. toctree::
    :hidden:

    geqrf
    geqrf_scratchpad_size
    gerqf
    gerqf_scratchpad_size
    getrf
    getrf_scratchpad_size
    getri
    getri_scratchpad_size
    getrs
    getrs_scratchpad_size
    hetrf
    hetrf_scratchpad_size
    orgqr
    orgqr_scratchpad_size
    ormqr
    ormqr_scratchpad_size
    ormrq
    ormrq_scratchpad_size
    potrf
    potrf_scratchpad_size
    potri
    potri_scratchpad_size
    potrs
    potrs_scratchpad_size
    sytrf
    sytrf_scratchpad_size
    trtrs
    trtrs_scratchpad_size
    ungqr
    ungqr_scratchpad_size
    unmqr
    unmqr_scratchpad_size
    unmrq
    unmrq_scratchpad_size
