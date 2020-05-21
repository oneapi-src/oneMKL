.. _onemkl_blas_gemm:

gemm
====


.. container::


   Computes a matrix-matrix product with general matrices.



      ``gemm`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``half`` 
         * -  ``float`` 
         * -  ``double`` 
         * -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 




.. container:: section


   .. rubric:: Description
      :class: sectiontitle


   The ``gemm`` routines compute a scalar-matrix-matrix product and add the
   result to a scalar-matrix product, with general matrices. The
   operation is defined as


  


      C <- alpha*op(A)*op(B) + beta*C


   where:


   ``op(X)`` is one of ``op(X) = X``, or ``op(X) = XT``, or
   ``op(X) = XH``,


   ``alpha`` and ``beta`` are scalars,


   ``A``, ``B`` and ``C`` are matrices:


   ``op(A)`` is an ``m``-by-``k`` matrix,


   ``op(B)`` is a ``k``-by-``n`` matrix,


   ``C`` is an ``m``-by-``n`` matrix.


gemm (Buffer Version)
---------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. cpp:function::  void onemkl::blas::gemm(sycl::queue &queue, transpose transa,      transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,      T alpha, sycl::buffer<T,1> &a, std::int64_t lda, sycl::buffer<T,1> &b,      std::int64_t ldb, T beta, sycl::buffer<T,1> &c, std::int64_t ldc)
.. container:: section


   .. rubric:: Input Parameters
      :class: sectiontitle


   queue
      The queue where the routine should be executed.


   transa
      Specifies the form of ``op(A)``, the transposition operation
      applied to ``A``.

   transb
      Specifies the form of ``op(B)``, the transposition operation
      applied to ``B``.


   m
      Specifies the number of rows of the matrix ``op(A)`` and of the
      matrix ``C``. The value of m must be at least zero.


   n
      Specifies the number of columns of the matrix ``op(B)`` and the
      number of columns of the matrix ``B``. The value of n must be at
      least zero.


   k
      Specifies the number of columns of the matrix ``op(A)`` and the
      number of rows of the matrix ``op(B)``. The value of k must be at
      least zero.


   alpha
      Scaling factor for the matrix-matrix product.


   a
      The buffer holding the input matrix ``A``. If ``A`` is not
      transposed, ``A`` is an ``m``-by-``k`` matrix so the array ``a``
      must have size at least ``lda``\ \*\ ``k``. If ``A`` is
      transposed, ``A`` is an ``k``-by-``m`` matrix so the array ``a``
      must have size at least ``lda``\ \*\ ``m``. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   lda
      The leading dimension of ``A``. Must be at least m if ``A`` is not
      transposed, and at least k if ``A`` is transposed. It must be
      positive.


   b
      The buffer holding the input matrix ``B``. If ``B`` is not
      transposed, ``B`` is an ``k``-by-``n`` matrix so the array ``b``
      must have size at least ``ldb``\ \*\ ``n``. If ``B`` is
      transposed, ``B`` is an ``n``-by-``k`` matrix so the array ``b``
      must have size at least ``ldb``\ \*\ ``k``. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   ldb
      The leading dimension of ``B``. Must be at least k if ``B`` is not
      transposed, and at least n if ``B`` is transposed. It must be
      positive.


   beta
      Scaling factor for matrix ``C``.


   c
      The buffer holding the input/output matrix ``C``. It must have a
      size of at least ldc\*n. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   ldc
      The leading dimension of ``C``. It must be positive and at least
      the size of m.


.. container:: section


   .. rubric:: Output Parameters
      :class: sectiontitle


   c
      The buffer, which is overwritten by
      ``alpha*op(A)*op(B) + beta*C``.


.. container:: section


   .. rubric:: Notes
      :class: sectiontitle


   If ``beta`` = 0, matrix ``C`` does not need to be initialized before
   calling ``gemm``.


gemm (USM Version)
------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. container:: dlsyntaxpara


         .. cpp:function::  sycl::event onemkl::blas::gemm(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n, std::int64_t k, T alpha, const T *a, std::int64_t lda, const T *b, std::int64_t ldb, T beta, T *c, std::int64_t ldc, const sycl::vector_class<sycl::event> &dependencies = {})
   .. container:: section


      .. rubric:: Input Parameters
         :class: sectiontitle


      queue
         The queue where the routine should be executed.


      transa
         Specifies the form of ``op(A)``, the transposition operation
         applied to ``A``.


      transb
         Specifies the form of ``op(B)``, the transposition operation
         applied to ``B``.


      m
         Specifies the number of rows of the matrix ``op(A)`` and of the
         matrix ``C``. The value of m must be at least zero.


      n
         Specifies the number of columns of the matrix ``op(B)`` and the
         number of columns of the matrix ``C``. The value of n must be
         at least zero.


      k
         Specifies the number of columns of the matrix ``op(A)`` and the
         number of rows of the matrix ``op(B)``. The value of k must be
         at least zero.


      alpha
         Scaling factor for the matrix-matrix product.


      a
         Pointer to input matrix ``A``. If ``A`` is not transposed,
         ``A`` is an ``m``-by-``k`` matrix so the array ``a`` must have
         size at least ``lda``\ \*\ ``k``. If ``A`` is transposed, ``A``
         is an ``k``-by-``m`` matrix so the array ``a`` must have size
         at least ``lda``\ \*\ ``m``. See `Matrix and Vector
         Storage <../matrix-storage.html>`__ for
         more details.


      lda
         The leading dimension of ``A``. Must be at least m if ``A`` is
         not transposed, and at least k if ``A`` is transposed. It must
         be positive.


      b
         Pointer to input matrix ``B``. If ``B`` is not transposed,
         ``B`` is an ``k``-by-``n`` matrix so the array ``b`` must have
         size at least ``ldb``\ \*\ ``n``. If ``B`` is transposed, ``B``
         is an ``n``-by-``k`` matrix so the array ``b`` must have size
         at least ``ldb``\ \*\ ``k``. See `Matrix and Vector
         Storage <../matrix-storage.html>`__ for
         more details.


      ldb
         The leading dimension of ``B``. Must be at least k if ``B`` is
         not transposed, and at least n if ``B`` is transposed. It must
         be positive.


      beta
         Scaling factor for matrix ``C``.


      c
         The pointer to input/output matrix ``C``. It must have a size
         of at least ldc\*n. See `Matrix and Vector
         Storage <../matrix-storage.html>`__ for
         more details.


      ldc
         The leading dimension of ``C``. It must be positive and at
         least the size of m.


      dependencies
         List of events to wait for before starting computation, if any.
         If omitted, defaults to no dependencies.


   .. container:: section


      .. rubric:: Output Parameters
         :class: sectiontitle


      c
         Pointer to the output matrix, overwritten by
         ``alpha*op(A)*op(B) + beta*C``.


   .. container:: section


      .. rubric:: Notes
         :class: sectiontitle


      If ``beta`` = 0, matrix ``C`` does not need to be initialized
      before calling ``gemm``.


   .. container:: section


      .. rubric:: Return Values
         :class: sectiontitle


      Output event to wait on to ensure computation is complete.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-3-routines`
