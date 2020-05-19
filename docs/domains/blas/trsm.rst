.. _onemkl_blas_trsm:

trsm
====


.. container::


   Solves a triangular matrix equation (forward or backward solve).



      ``trsm`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``float`` 
         * -  ``double`` 
         * -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 




.. container:: section


   .. rubric:: Description
      :class: sectiontitle


   The ``trsm`` routines solve one of the following matrix equations:



      op(A)*X = alpha*B,


   or


      X*op(A) = alpha*B,


   where:


   op(``A``) is one of op(``A``) = ``A``, or op(``A``) =
   ``A``\ :sup:`T`, or op(``A``) = ``A``\ :sup:`H`,


   ``alpha`` is a scalar,


   ``A`` is a triangular matrix, and


   ``B`` and ``X`` are ``m`` x ``n`` general matrices.


   ``A`` is either ``m`` x ``m`` or ``n`` x ``n``, depending on whether
   it multiplies ``X`` on the left or right. On return, the matrix ``B``
   is overwritten by the solution matrix ``X``.


trsm (Buffer Version)
---------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. cpp:function::  void onemkl::blas::trsm(sycl::queue &queue, side left_right, uplo upper_lower, transpose transa, diag unit_diag, std::int64_t m, std::int64_t n, T alpha, sycl::buffer<T,1> &a, std::int64_t lda, sycl::buffer<T,1> &b, std::int64_t ldb)
.. container:: section


   .. rubric:: Input Parameters
      :class: sectiontitle


   queue
      The queue where the routine should be executed.


   left_right
      Specifies whether ``A`` multiplies ``X`` on the left
      (``side::left``) or on the right (``side::right``). See :ref:`onemkl_datatypes` for more details.


   uplo
      Specifies whether the matrix ``A`` is upper or lower triangular. See :ref:`onemkl_datatypes` for more details.


   trans
      Specifies op(``A``), the transposition operation applied to ``A``. See :ref:`onemkl_datatypes` for more details.


   unit_diag
      Specifies whether ``A`` is assumed to be unit triangular (all
      diagonal elements are 1). See :ref:`onemkl_datatypes` for more details.


   m
      Specifies the number of rows of ``B``. The value of ``m`` must be
      at least zero.


   n
      Specifies the number of columns of ``B``. The value of ``n`` must
      be at least zero.


   alpha
      Scaling factor for the solution.


   a
      Buffer holding input matrix ``A``. Must have size at least
      ``lda``\ \*\ ``m`` if ``left_right`` = ``side::left``, or
      ``lda``\ \*\ ``n`` if ``left_right`` = ``side::right``. See
      `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   lda
      Leading dimension of ``A``. Must be at least ``m`` if
      ``left_right`` = ``side::left``, and at least ``n`` if
      ``left_right`` = ``side::right``. Must be positive.


   b
      Buffer holding input/output matrix ``B``. Must have size at least
      ``ldb``\ \*\ ``n``. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   ldb
      Leading dimension of ``B``. Must be at least ``m`` and positive.


.. container:: section


   .. rubric:: Output Parameters
      :class: sectiontitle


   b
      Output buffer. Overwritten by the solution matrix ``X``.


.. container:: section


   .. rubric:: Notes
      :class: sectiontitle


   If ``alpha`` = 0, matrix ``B`` is set to zero, and ``A`` and ``B`` do
   not need to be initialized at entry.


trsm (USM Version)
------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. container:: dlsyntaxpara


         .. cpp:function::  sycl::event onemkl::blas::trsm(sycl::queue &queue, side left_right, uplo upper_lower, transpose transa, diag unit_diag, std::int64_t m, std::int64_t n, T alpha, const T* a, std::int64_t lda, T* b, std::int64_t ldb, const sycl::vector_class<sycl::event> &dependencies = {})
   .. container:: section


      .. rubric:: Input Parameters
         :class: sectiontitle


      queue
         The queue where the routine should be executed.


      left_right
         Specifies whether ``A`` multiplies ``X`` on the left
         (``side::left``) or on the right (``side::right``). See :ref:`onemkl_datatypes` for more details.


      uplo
         Specifies whether the matrix ``A`` is upper or lower
         triangular. See :ref:`onemkl_datatypes` for more details.


      transa
         Specifies op(``A``), the transposition operation applied to
         ``A``. See :ref:`onemkl_datatypes` for more details.


      unit_diag
         Specifies whether ``A`` is assumed to be unit triangular (all
         diagonal elements are 1). See :ref:`onemkl_datatypes` for more details.


      m
         Specifies the number of rows of ``B``. The value of ``m`` must
         be at least zero.


      n
         Specifies the number of columns of ``B``. The value of ``n``
         must be at least zero.


      alpha
         Scaling factor for the solution.


      a
         Pointer to input matrix ``A``. Must have size at least
         ``lda``\ \*\ ``m`` if ``left_right`` = ``side::left``, or
         ``lda``\ \*\ ``n`` if ``left_right`` = ``side::right``. See
         `Matrix and Vector
         Storage <../matrix-storage.html>`__ for
         more details.


      lda
         Leading dimension of ``A``. Must be at least ``m`` if
         ``left_right`` = ``side::left``, and at least ``n`` if
         ``left_right`` = ``side::right``. Must be positive.


      b
         Pointer to input/output matrix ``B``. Must have size at least
         ``ldb``\ \*\ ``n``. See `Matrix and Vector
         Storage <../matrix-storage.html>`__ for
         more details.


      ldb
         Leading dimension of ``B``. Must be at least ``m`` and
         positive.


      dependencies
         List of events to wait for before starting computation, if any.
         If omitted, defaults to no dependencies.


   .. container:: section


      .. rubric:: Output Parameters
         :class: sectiontitle


      b
         Pointer to the output matrix. Overwritten by the solution
         matrix ``X``.


   .. container:: section


      .. rubric:: Notes
         :class: sectiontitle


      If ``alpha`` = 0, matrix ``B`` is set to zero, and ``A`` and ``B``
      do not need to be initialized at entry.


   .. container:: section


      .. rubric:: Return Values
         :class: sectiontitle


      Output event to wait on to ensure computation is complete.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-3-routines`
