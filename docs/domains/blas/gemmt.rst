.. _onemkl_blas_gemmt:

gemmt
=====


.. container::


   Computes a matrix-matrix product with general matrices, but updates
   only the upper or lower triangular part of the result matrix.



         ``gemmt`` supports the following precisions.


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


      The gemmt routines compute a scalar-matrix-matrix product and add
      the result to the upper or lower part of a scalar-matrix product,
      with general matrices. The operation is defined as:


      ::


         C <- alpha*op(A)*op(B) + beta*C 


      where:


      op(X) is one of op(X) = X, or op(X) = X\ :sup:`T`, or op(X) = X\ :sup:`H`


      ``alpha`` and ``beta`` are scalars


      ``A``, ``B``, and ``C`` are matrices


      op(``A``) is ``n`` x ``k``, op(``B``) is ``k`` x ``n``, and
      ``C`` is ``n`` x ``n``.


gemmt (Buffer Version)
----------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. container:: dlsyntaxpara


         .. cpp:function::  void onemkl::blas::gemmt(sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, std::int64_t n, std::int64_t k, T alpha, sycl::buffer<T,1> &a, std::int64_t lda, sycl::buffer<T,1> &b, std::int64_t ldb, T beta, sycl::buffer<T,1> &c, std::int64_t ldc)
   .. container:: section


      .. rubric:: Input Parameters
         :class: sectiontitle


      queue
         The queue where the routine should be executed.


      upper_lower
         Specifies whether ``C``\ ’s data is stored in its upper or
         lower triangle. See :ref:`onemkl_datatypes` for more details.

      
      transa
         Specifies op(``A``), the transposition operation applied to
         ``A``. See :ref:`onemkl_datatypes` for more details.


      transb
         Specifies op(``B``), the transposition operation applied to
         ``B``. See :ref:`onemkl_datatypes` for more details.


      n
         Number of columns of op(``A``), columns of op(``B``), and
         columns of\ ``C``. Must be at least zero.


      k
         Number of columns of op(``A``) and rows of op(``B``). Must be
         at least zero.


      alpha
         Scaling factor for the matrix-matrix product.


      a
         Buffer holding the input matrix ``A``.


         If ``A`` is not transposed, ``A`` is an ``n``-by-``k`` matrix
         so the array ``a`` must have size at least ``lda``\ \*\ ``k``.


         If ``A`` is transposed, ``A`` is a ``k``-by-``n`` matrix so the
         array ``a`` must have size at least ``lda``\ \*\ ``n``.


         See `Matrix Storage <../matrix-storage.html>`__ for more details.


      lda
         Leading dimension of ``A``. Must be at least ``n`` if ``A`` is
         not transposed, and at least ``k`` if ``A`` is transposed. Must
         be positive.


      b
         Buffer holding the input matrix ``B``.


         If ``B`` is not transposed, ``B`` is a ``k``-by-``n`` matrix so
         the array ``b`` must have size at least ``ldb``\ \*\ ``n``.


         If ``B`` is transposed, ``B`` is an ``n``-by-``k`` matrix so
         the array ``b`` must have size at least ``ldb``\ \*\ ``k``.


         See `Matrix Storage <../matrix-storage.html>`__ for more details.


      ldb
         Leading dimension of ``B``. Must be at least ``k`` if ``B`` is
         not transposed, and at least ``n`` if ``B`` is transposed. Must
         be positive.


      beta
         Scaling factor for matrix ``C``.


      c
         Buffer holding the input/output matrix ``C``. Must have size at
         least ``ldc`` \* ``n``. See `Matrix
         Storage <../matrix-storage.html>`__ for
         more details.


      ldc
         Leading dimension of ``C``. Must be positive and at least
         ``m``.


   .. container:: section


      .. rubric:: Output Parameters
         :class: sectiontitle


      c
         Output buffer, overwritten by the upper or lower triangular
         part of alpha\*op(``A``)*op(``B``) + beta\*\ ``C``.


   .. container:: section


      .. rubric:: Notes
         :class: sectiontitle


      If ``beta`` = 0, matrix ``C`` does not need to be initialized
      before calling gemmt.


gemmt (USM Version)
-------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. container:: dlsyntaxpara


         .. cpp:function::  sycl::event onemkl::blas::gemmt(sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, std::int64_t n, std::int64_t k, T alpha, const T* a, std::int64_t lda, const T* b, std::int64_t ldb, T beta, T* c, std::int64_t ldc, const sycl::vector_class<sycl::event> &dependencies = {})
   .. container:: section


      .. rubric:: Input Parameters
         :class: sectiontitle


      queue
         The queue where the routine should be executed.


      upper_lower
         Specifies whether ``C``\ ’s data is stored in its upper or
         lower triangle. See
         :ref:`onemkl_datatypes` for
         more details.



      transa
         Specifies op(``A``), the transposition operation applied to
         ``A``. See
         :ref:`onemkl_datatypes` for
         more details.



      transb
         Specifies op(``B``), the transposition operation applied to
         ``B``. See
         :ref:`onemkl_datatypes` for
         more details.

 

      n
         Number of columns of op(``A``), columns of op(``B``), and
         columns of\ ``C``. Must be at least zero.


      k
         Number of columns of op(``A``) and rows of op(``B``). Must be
         at least zero.


      alpha
         Scaling factor for the matrix-matrix product.


      a
         Pointer to input matrix ``A``.


         If ``A`` is not transposed, ``A`` is an ``n``-by-``k`` matrix
         so the array ``a`` must have size at least ``lda``\ \*\ ``k``.


         If ``A`` is transposed, ``A`` is a ``k``-by-``n`` matrix so the
         array ``a`` must have size at least ``lda``\ \*\ ``n``.


         See `Matrix
         Storage <../matrix-storage.html>`__ for
         more details.


      lda
         Leading dimension of ``A``. Must be at least ``n`` if ``A`` is
         not transposed, and at least ``k`` if ``A`` is transposed. Must
         be positive.


      b
         Pointer to input matrix ``B``.


         If ``B`` is not transposed, ``B`` is a ``k``-by-``n`` matrix so
         the array ``b`` must have size at least ``ldb``\ \*\ ``n``.


         If ``B`` is transposed, ``B`` is an ``n``-by-``k`` matrix so
         the array ``b`` must have size at least ``ldb``\ \*\ ``k``.


         See `Matrix
         Storage <../matrix-storage.html>`__ for
         more details.


      ldb
         Leading dimension of ``B``. Must be at least ``k`` if ``B`` is
         not transposed, and at least ``n`` if ``B`` is transposed. Must
         be positive.


      beta
         Scaling factor for matrix ``C``.


      c
         Pointer to input/output matrix ``C``. Must have size at least
         ``ldc`` \* ``n``. See `Matrix
         Storage <../matrix-storage.html>`__ for
         more details.


      ldc
         Leading dimension of ``C``. Must be positive and at least
         ``m``.


      dependencies
         List of events to wait for before starting computation, if any.
         If omitted, defaults to no dependencies.


   .. container:: section


      .. rubric:: Output Parameters
         :class: sectiontitle


      c
         Pointer to the output matrix, overwritten by the upper or lower
         triangular part of alpha\*op(``A``)*op(``B``) + beta\*\ ``C``.


   .. container:: section


      .. rubric:: Notes
         :class: sectiontitle


      If ``beta`` = 0, matrix ``C`` does not need to be initialized
      before calling gemmt.


   .. container:: section


      .. rubric:: Return Values
         :class: sectiontitle


      Output event to wait on to ensure computation is complete.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-like-extensions`
