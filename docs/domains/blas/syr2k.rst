.. _onemkl_blas_syr2k:

syr2k
=====


.. container::


   Performs a symmetric rank-2k update.



      ``syr2k`` supports the following precisions:


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


   The ``syr2k`` routines perform a rank-2k update of an ``n`` x ``n``
   symmetric matrix ``C`` by general matrices ``A`` and ``B``. If
   ``trans`` = ``transpose::nontrans``, the operation is defined as:


      C <- alpha*(A*B :sup:`T` + B*A :sup:`T`) + beta*C


   where ``A`` is ``n`` x ``k`` and ``B`` is ``k`` x ``n``.


   If ``trans`` = ``transpose::trans``, the operation is defined as:


      C <- alpha*(A :sup:`T` * B + B :sup:`T` * A) + beta * C


   where ``A`` is ``k`` x ``n`` and ``B`` is ``n`` x ``k``.


   In both cases:


   ``alpha`` and ``beta`` are scalars,


   ``C`` is a symmetric matrix and ``A``,\ ``B`` are general matrices,


   The inner dimension of both matrix multiplications is ``k``.


syr2k (Buffer Version)
----------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. cpp:function::  void onemkl::blas::syr2k(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k, T alpha, sycl::buffer<T,1> &a, std::int64_t lda, sycl::buffer<T,1> &b, std::int64_t ldb, T beta, sycl::buffer<T,1> &c, std::int64_t ldc)

.. container:: section


   .. rubric:: Input Parameters
      :class: sectiontitle


   queue
      The queue where the routine should be executed.


   upper_lower
      Specifies whether ``A``'s data is stored in its upper or lower
      triangle. See :ref:`onemkl_datatypes` for more details.


   trans
      Specifies the operation to apply, as described above. Conjugation
      is never performed, even if ``trans`` = ``transpose::conjtrans``.


   n
      Number of rows and columns in ``C``.The value of ``n`` must be at
      least zero.


   k
      Inner dimension of matrix multiplications.The value of ``k`` must
      be at least zero.


   alpha
      Scaling factor for the rank-2\ ``k`` update.


   a
      Buffer holding input matrix ``A``. If ``A`` is not transposed,
      ``A`` is an ``m``-by-``k`` matrix so the array ``a`` must have
      size at least ``lda``\ \*\ ``k``. If ``A`` is transposed, ``A`` is
      an ``k``-by-``m`` matrix so the array ``a`` must have size at
      least ``lda``\ \*\ ``m``. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   lda
      Leading dimension of ``A``. Must be at least ``n`` if ``trans`` =
      ``transpose::nontrans``, and at least ``k`` otherwise. Must be
      positive.


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


   beta
      Scaling factor for matrix ``C``.


   c
      Buffer holding input/output matrix ``C``. Must have size at least
      ``ldc``\ \*\ ``n``. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details


   ldc
      Leading dimension of ``C``. Must be positive and at least ``n``.


.. container:: section


   .. rubric:: Output Parameters
      :class: sectiontitle


   c
      Output buffer, overwritten by the updated ``C`` matrix.


syr2k (USM Version)
-------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. container:: dlsyntaxpara


         .. cpp:function::  sycl::event onemkl::blas::syr2k(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k, T alpha, const T* a, std::int64_t lda, const T* b, std::int64_t ldb, T beta, T* c, std::int64_t ldc, const sycl::vector_class<sycl::event> &dependencies = {})
   .. container:: section


      .. rubric:: Input Parameters
         :class: sectiontitle


      queue
         The queue where the routine should be executed.


      upper_lower
         Specifies whether ``A``'s data is stored in its upper or lower
         triangle. See :ref:`onemkl_datatypes` for more details.


      trans
         Specifies the operation to apply, as described above.
         Conjugation is never performed, even if ``trans`` =
         ``transpose::conjtrans``.


      n
         Number of rows and columns in ``C``. The value of ``n`` must be
         at least zero.


      k
         Inner dimension of matrix multiplications.The value of ``k``
         must be at least zero.


      alpha
         Scaling factor for the rank-2\ ``k`` update.


      a
         Pointer to input matrix ``A``. If ``A`` is not transposed,
         ``A`` is an ``m``-by-``k`` matrix so the array ``a`` must have
         size at least ``lda``\ \*\ ``k``. If ``A`` is transposed, ``A``
         is an ``k``-by-``m`` matrix so the array ``a`` must have size
         at least ``lda``\ \*\ ``m``. See `Matrix and Vector
         Storage <../matrix-storage.html>`__ for
         more details.


      lda
         Leading dimension of ``A``. Must be at least ``n`` if ``trans``
         = ``transpose::nontrans``, and at least ``k`` otherwise. Must
         be positive.


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


      beta
         Scaling factor for matrix ``C``.


      c
         Pointer to input/output matrix ``C``. Must have size at least
         ``ldc``\ \*\ ``n``. See `Matrix and Vector
         Storage <../matrix-storage.html>`__ for
         more details


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
