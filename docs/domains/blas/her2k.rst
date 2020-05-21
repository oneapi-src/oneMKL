.. _onemkl_blas_her2k:

her2k
=====


.. container::


   Performs a Hermitian rank-2k update.



      ``her2k`` supports the following precisions:


      .. list-table:: 
         :header-rows: 1

         * -  T 
           -  T_real 
         * -  ``std::complex<float>`` 
           -  ``float`` 
         * -  ``std::complex<double>`` 
           -  ``double`` 




.. container:: section


   .. rubric:: Description
      :class: sectiontitle


   The ``her2k`` routines perform a rank-2k update of an ``n`` x ``n``
   Hermitian matrix ``C`` by general matrices ``A`` and ``B``. If
   ``trans`` = ``transpose::nontrans``. The operation is defined as


  


      C <- alpha*A*B :sup:`H` + conjg(alpha)*B*A :sup:`H` + beta*C


   where ``A`` is ``n`` x ``k`` and ``B`` is ``k`` x ``n``.


   If ``trans`` = ``transpose::conjtrans``, the operation is defined as:


  


      C <- alpha*B*A :sup:`H` + conjg(alpha)*A*B :sup:`H` + beta*C


   where ``A`` is ``k`` x ``n`` and ``B`` is ``n`` x ``k``.


   In both cases:


   ``alpha`` is a complex scalar and ``beta`` is a real scalar.


   ``C`` is a Hermitian matrix and ``A, B`` are general matrices.


   The inner dimension of both matrix multiplications is ``k``.


her2k (Buffer Version)
----------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. cpp:function::  void onemkl::blas::her2k(sycl::queue &queue, uplo upper_lower,      transpose trans, std::int64_t n, std::int64_t k, T alpha,      sycl::buffer<T,1> &a, std::int64_t lda, sycl::buffer<T,1> &b, std::int64_t      ldb, T_real beta, sycl::buffer<T,1> &c, std::int64_t ldc)
.. container:: section


   .. rubric:: Input Parameters
      :class: sectiontitle


   queue
      The queue where the routine should be executed.


   upper_lower
      Specifies whether ``A``'s data is stored in its upper or lower
      triangle. See :ref:`onemkl_datatypes` for more details.



   trans
      Specifies the operation to apply, as described above. Supported
      operations are ``transpose::nontrans`` and
      ``transpose::conjtrans``.


   n
      The number of rows and columns in ``C``. The value of ``n`` must
      be at least zero.


   k
      The inner dimension of matrix multiplications. The value of ``k``
      must be at least equal to zero.


   alpha
      Complex scaling factor for the rank-2\ ``k`` update.


   a
      Buffer holding input matrix ``A``. If ``trans`` =
      ``transpose::nontrans``, ``A`` is an ``n``-by-``k`` matrix so the
      array ``a`` must have size at least ``lda``\ \*\ ``k``. Otherwise,
      ``A`` is an ``k``-by-``n`` matrix so the array ``a`` must have
      size at least ``lda``\ \*\ ``n``. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   lda
      Leading dimension of ``A``. Must be at least ``n`` if ``trans`` =
      ``transpose::nontrans``, and at least ``k`` otherwise. Must be
      positive.


   beta
      Real scaling factor for matrix ``C``.


   b
      Buffer holding input matrix ``B``. If ``trans`` =
      ``transpose::nontrans``, ``B`` is an ``k``-by-``n`` matrix so the
      array ``b`` must have size at least ``ldb``\ \*\ ``n``. Otherwise,
      ``B`` is an ``n``-by-``k`` matrix so the array ``b`` must have
      size at least ``ldb``\ \*\ ``k``. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   ldb
      Leading dimension of ``B``. Must be at least ``k`` if ``trans`` =
      ``transpose::nontrans``, and at least ``n`` otherwise. Must be
      positive.


   c
      Buffer holding input/output matrix ``C``. Must have size at least
      ``ldc``\ \*\ ``n``. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   ldc
      Leading dimension of ``C``. Must be positive and at least ``n``.


.. container:: section


   .. rubric:: Output Parameters
      :class: sectiontitle


   c
      Output buffer, overwritten by the updated ``C`` matrix.


her2k (USM Version)
-------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. container:: dlsyntaxpara


         .. cpp:function::  sycl::event onemkl::blas::her2k(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k, T alpha, const T* a, std::int64_t lda, const T* b, std::int64_t ldb, T_real beta, T* c, std::int64_t ldc, const sycl::vector_class<sycl::event> &dependencies = {})
   .. container:: section


      .. rubric:: Input Parameters
         :class: sectiontitle


      queue
         The queue where the routine should be executed.


      upper_lower
         Specifies whether ``A``'s data is stored in its upper or lower
         triangle. See :ref:`onemkl_datatypes` for more details.



      trans
         Specifies the operation to apply, as described above. Supported
         operations are ``transpose::nontrans`` and
         ``transpose::conjtrans``.


      n
         The number of rows and columns in ``C``. The value of ``n``
         must be at least zero.


      k
         The inner dimension of matrix multiplications. The value of
         ``k`` must be at least equal to zero.


      alpha
         Complex scaling factor for the rank-2\ ``k`` update.


      a
         Pointer to input matrix ``A``. If ``trans`` =
         ``transpose::nontrans``, ``A`` is an ``n``-by-``k`` matrix so
         the array ``a`` must have size at least ``lda``\ \*\ ``k``.
         Otherwise, ``A`` is an ``k``-by-``n`` matrix so the array ``a``
         must have size at least ``lda``\ \*\ ``n``. See `Matrix and
         Vector
         Storage <../matrix-storage.html>`__ for
         more details.


      lda
         Leading dimension of ``A``. Must be at least ``n`` if ``trans``
         = ``transpose::nontrans``, and at least ``k`` otherwise. Must
         be positive.


      beta
         Real scaling factor for matrix ``C``.


      b
         Pointer to input matrix ``B``. If ``trans`` =
         ``transpose::nontrans``, ``B`` is an ``k``-by-``n`` matrix so
         the array ``b`` must have size at least ``ldb``\ \*\ ``n``.
         Otherwise, ``B`` is an ``n``-by-``k`` matrix so the array ``b``
         must have size at least ``ldb``\ \*\ ``k``. See `Matrix and
         Vector
         Storage <../matrix-storage.html>`__ for
         more details.


      ldb
         Leading dimension of ``B``. Must be at least ``k`` if ``trans``
         = ``transpose::nontrans``, and at least ``n`` otherwise. Must
         be positive.


      c
         Pointer to input/output matrix ``C``. Must have size at least
         ``ldc``\ \*\ ``n``. See `Matrix and Vector
         Storage <../matrix-storage.html>`__ for
         more details.


      ldc
         Leading dimension of ``C``. Must be positive and at least
         ``n``.


      dependencies
         List of events to wait for before starting computation, if any.
         If omitted, defaults to no dependencies.


   .. container:: section


      .. rubric:: Output Parameters
         :class: sectiontitle


      c
         Pointer to the output matrix, overwritten by the updated ``C``
         matrix.


   .. container:: section


      .. rubric:: Return Values
         :class: sectiontitle


      Output event to wait on to ensure computation is complete.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-3-routines`
