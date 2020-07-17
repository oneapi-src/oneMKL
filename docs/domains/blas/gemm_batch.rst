.. _onemkl_blas_gemm_batch:

gemm_batch
==========


.. container::

   The ``gemm_batch`` routines are batched versions of `gemm <gemm.html>`__, performing
   multiple ``gemm`` operations in a single call. Each ``gemm`` 
   operation perform a matrix-matrix product with general matrices.
   
  
      ``gemm_batch`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``float`` 
         * -  ``double`` 
         * -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 


gemm_batch (Buffer Version)
---------------------------

.. container:: section


   .. rubric:: Description
      :class: sectiontitle


   The buffer version of ``gemm_batch`` supports only the strided API. 
   
   The strided API operation is defined as


   ::


      for i = 0 … batch_size – 1
          A, B and C are matrices at offset i * stridea, i * strideb, i * stridec in a, b and c.
          C := alpha * op(A) * op(B) + beta * C
      end for


   where:


   op(X) is one of op(X) = X, or op(X) = X\ :sup:`T`, or op(X) = X\ :sup:`H`


   ``alpha`` and ``beta`` are scalars


   ``A``, ``B``, and ``C`` are matrices

   op(``A``) is ``m``\ ``x``\ ``k``, op(``B``) is 
   ``k``\ ``x``\ ``n``, and ``C`` is ``m``\ ``x``\ ``n``.

   The a, b and c buffers contain all the input matrices. The stride 
   between matrices is given by the stride parameter. The total number
   of matrices in a, b and c buffers is given by the ``batch_size`` parameter.

   **Strided API**

.. container:: section


   .. rubric:: Syntax
      :class: sectiontitle


   .. cpp:function::  void oneapi::mkl::blas::gemm_batch(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n, std::int64_t k, T alpha, sycl::buffer<T,1> &a, std::int64_t lda, std::int64_t stridea, sycl::buffer<T,1> &b, std::int64_t ldb, std::int64_t strideb, T beta, sycl::buffer<T,1> &c, std::int64_t ldc, std::int64_t stridec, std::int64_t batch_size)


.. container:: section


   .. rubric:: Input Parameters
      :class: sectiontitle


   queue
      The queue where the routine should be executed.


   transa
      Specifies ``op(A)`` the transposition operation applied to the
      matrices ``A``. See :ref:`onemkl_datatypes` for more details.

   transb
      Specifies ``op(B)`` the transposition operation applied to the
      matrices ``B``. See :ref:`onemkl_datatypes` for more details.

   m
      Number of rows of ``op(A)`` and ``C``. Must be at least zero.


   n
      Number of columns of ``op(B)`` and ``C``. Must be at least zero.


   k
      Number of columns of ``op(A)`` and rows of ``op(B)``. Must be at
      least zero.


   alpha
      Scaling factor for the matrix-matrix products.


   a
      Buffer holding the input matrices ``A`` with size ``stridea*batch_size``.


   lda
      Leading dimension of the matrices ``A``. Must be at least ``m`` if
      the matrices ``A`` are not transposed, and at least ``k`` if the
      matrices ``A`` are transposed. Must be positive.


   stridea
      Stride between different ``A`` matrices.


   b
      Buffer holding the input matrices ``B`` with size ``strideb*batch_size``.


   ldb
      Leading dimension of the matrices ``B``. Must be at least ``k`` if
      the matrices ``B`` are not transposed, and at least ``n`` if the
      matrices ``B`` are transposed. Must be positive.


   strideb
      Stride between different ``B`` matrices.


   beta
      Scaling factor for the matrices ``C``.


   c
      Buffer holding input/output matrices ``C`` with size ``stridec*batch_size``.


   ldc
      Leading dimension of ``C``. Must be positive and at least ``m``.


   stridec
      Stride between different ``C`` matrices. Must be at least
      ``ldc*n``.


   batch_size
      Specifies the number of matrix multiply operations to perform.


.. container:: section


   .. rubric:: Output Parameters
      :class: sectiontitle


   c
      Output buffer, overwritten by ``batch_size`` matrix multiply
      operations of the form\ ``alpha*op(A)*op(B) + beta*C``.


.. container:: section


   .. rubric:: Notes
      :class: sectiontitle


   If ``beta`` = 0, matrix ``C`` does not need to be initialized before
   calling ``gemm_batch``.


gemm_batch (USM Version)
---------------------------

.. container:: section

   .. rubric:: Description
      :class: sectiontitle


   The USM version of ``gemm_batch`` supports the group API and strided API. 

   The group API operation is defined as


   ::


      idx = 0
      for i = 0 … group_count – 1
          for j = 0 … group_size – 1
              A, B, and C are matrices in a[idx], b[idx] and c[idx]
              C := alpha[i] * op(A) * op(B) + beta[i] * C
              idx = idx + 1
          end for
      end for


   The strided API operation is defined as


   ::


      for i = 0 … batch_size – 1
          A, B and C are matrices at offset i * stridea, i * strideb, i * stridec in a, b and c.
          C := alpha * op(A) * op(B) + beta * C
      end for


   where:


   op(X) is one of op(X) = X, or op(X) = X\ :sup:`T`, or op(X) = X\ :sup:`H`


   ``alpha`` and ``beta`` are scalars


   ``A``, ``B``, and ``C`` are matrices
   
   op(``A``) is ``m``\ ``x``\ ``k``, op(``B``) is ``k``\ ``x``\ ``n``, and ``C`` is ``m``\ ``x``\ ``n``.

    
   For group API, a, b and c arrays contain the pointers for all the input matrices. 
   The total number of matrices in a, b and c are given by: 
    
      total_batch_count = sum of all of the group_size entries    
    
    
   For strided API, a, b, c arrays contain all the input matrices. The total number of matrices 
   in a, b and c are given by the ``batch_size`` parameter.  
      
   **Group API**

.. container:: section


   .. rubric:: Syntax
      :class: sectiontitle


   .. container:: dlsyntaxpara
   
      .. cpp:function::  sycl::event oneapi::mkl::blas::gemm_batch(sycl::queue &queue, transpose *transa, transpose *transb, std::int64_t *m, std::int64_t *n, std::int64_t *k, T *alpha, const T **a, std::int64_t *lda, const T **b, std::int64_t *ldb, T *beta, T **c, std::int64_t *ldc, std::int64_t group_count, std::int64_t *group_size, const sycl::vector_class<sycl::event> &dependencies = {})


.. container:: section


   .. rubric:: Input Parameters
      :class: sectiontitle


   queue
      The queue where the routine should be executed.


   transa
      Array of ``group_count`` ``oneapi::mkl::transpose`` values. ``transa[i]`` specifies the form of ``op(A)`` used in
      the matrix multiplication in group ``i``. See :ref:`onemkl_datatypes` for more details.


   transb
      Array of ``group_count`` ``oneapi::mkl::transpose`` values. ``transb[i]`` specifies the form of ``op(B)`` used in
      the matrix multiplication in group ``i``. See :ref:`onemkl_datatypes` for more details.


   m
      Array of ``group_count`` integers. ``m[i]`` specifies the
      number of rows of ``op(A)`` and ``C`` for every matrix in group ``i``. All entries must be at least zero.


   n
      Array of ``group_count`` integers. ``n[i]`` specifies the
      number of columns of ``op(B)`` and ``C`` for every matrix in group ``i``. All entries must be at least zero.


   k
      Array of ``group_count`` integers. ``k[i]`` specifies the
      number of columns of ``op(A)`` and rows of ``op(B)`` for every matrix in group ``i``. All entries must be at
      least zero.


   alpha
      Array of ``group_count`` scalar elements. ``alpha[i]`` specifies the scaling factor for every matrix-matrix
      product in group ``i``.


   a
      Array of pointers to input matrices ``A`` with size ``total_batch_count``. 
      
      See `Matrix Storage <../matrix-storage.html>`__ for more details.


   lda
      Array of ``group_count`` integers. ``lda[i]`` specifies the leading dimension of ``A`` for every matrix in group ``i``. 
      All entries must be at least ``m``
      if ``A`` is not transposed, and at least ``k`` if ``A`` is
      transposed. All entries must be positive.


   b
      Array of pointers to input matrices ``B`` with size ``total_batch_count``. 
      
      See `Matrix Storage <../matrix-storage.html>`__ for more details.


   ldb
      Array of ``group_count`` integers. ``ldb[i]`` specifies the leading dimension of ``B`` for every matrix in group ``i``. 
      All entries must be at least ``k``
      if ``B`` is not transposed, and at least ``n`` if ``B`` is
      transposed. All entries must be positive.


   beta
      Array of ``group_count`` scalar elements. ``beta[i]`` specifies the scaling factor for matrix ``C`` 
      for every matrix in group ``i``.


   c
      Array of pointers to input/output matrices ``C`` with size ``total_batch_count``. 
      
      See `Matrix Storage <../matrix-storage.html>`__ for more details.


   ldc
      Array of ``group_count`` integers. ``ldc[i]`` specifies the leading dimension of ``C`` for every matrix in group ``i``. 
      All entries must be positive and at least ``m``.


   group_count
      Specifies the number of groups. Must be at least 0.


   group_size
      Array of ``group_count`` integers. ``group_size[i]`` specifies the
      number of matrix multiply products in group ``i``. All entries must be at least 0.


   dependencies
         List of events to wait for before starting computation, if any.
         If omitted, defaults to no dependencies.


.. container:: section


   .. rubric:: Output Parameters
      :class: sectiontitle


   c
      Overwritten by the ``m[i]``-by-``n[i]`` matrix calculated by 
      ``(alpha[i]*op(A)*op(B) + beta[i]*C)`` for group ``i``.



   .. container:: section


      .. rubric:: Notes
         :class: sectiontitle


      If ``beta`` = 0, matrix ``C`` does not need to be initialized
      before calling ``gemm_batch``.


   .. container:: section


      .. rubric:: Return Values
         :class: sectiontitle


      Output event to wait on to ensure computation is complete.




   **Strided API**

.. container:: section


   .. rubric:: Syntax
      :class: sectiontitle

   .. container:: dlsyntaxpara

      .. cpp:function::  sycl::event oneapi::mkl::blas::gemm_batch(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n, std::int64_t k, T alpha, const T *a, std::int64_t lda, std::int64_t stridea, const T *b, std::int64_t ldb, std::int64_t strideb, T beta, T *c, std::int64_t ldc, std::int64_t stridec, std::int64_t batch_size, const sycl::vector_class<sycl::event> &dependencies = {})


.. container:: section


   .. rubric:: Input Parameters
      :class: sectiontitle


   queue
      The queue where the routine should be executed.


   transa
      Specifies ``op(A)`` the transposition operation applied to the
      matrices ``A``. See :ref:`onemkl_datatypes` for more details.



   transb
      Specifies ``op(B)`` the transposition operation applied to the
      matrices ``B``. See :ref:`onemkl_datatypes` for more details.


   m
      Number of rows of ``op(A)`` and ``C``. Must be at least zero.


   n
      Number of columns of ``op(B)`` and ``C``. Must be at least zero.


   k
      Number of columns of ``op(A)`` and rows of ``op(B)``. Must be at
      least zero.


   alpha
      Scaling factor for the matrix-matrix products.


   a
      Pointer to input matrices ``A`` with size ``stridea*batch_size``.


   lda
      Leading dimension of the matrices ``A``. Must be at least ``m`` if
      the matrices ``A`` are not transposed, and at least ``k`` if the
      matrices ``A`` are transposed. Must be positive.


   stridea
      Stride between different ``A`` matrices.


   b
      Pointer to input matrices ``B`` with size ``strideb*batch_size``.


   ldb
      Leading dimension of the matrices ``B``. Must be at least ``k`` if
      the matrices ``B`` are not transposed, and at least ``n`` if the
      matrices ``B`` are transposed. Must be positive.


   strideb
      Stride between different ``B`` matrices.



   beta
      Scaling factor for the matrices ``C``.


   c
      Pointer to input/output matrices ``C`` with size ``stridec*batch_size``.


   ldc
      Leading dimension of ``C``. Must be positive and at least ``m``.


   stridec
      Stride between different ``C`` matrices.


   batch_size
      Specifies the number of matrix multiply operations to perform.


   dependencies
         List of events to wait for before starting computation, if any.
         If omitted, defaults to no dependencies.


.. container:: section


   .. rubric:: Output Parameters
      :class: sectiontitle


   c
      Output matrices, overwritten by ``batch_size`` matrix multiply
      operations of the form ``alpha*op(A)*op(B) + beta*C``.


.. container:: section


   .. rubric:: Notes
      :class: sectiontitle


   If ``beta`` = 0, matrix ``C`` does not need to be initialized before
   calling ``gemm_batch``.


.. container:: section


      .. rubric:: Return Values
         :class: sectiontitle


      Output event to wait on to ensure computation is complete.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-like-extensions`
      

