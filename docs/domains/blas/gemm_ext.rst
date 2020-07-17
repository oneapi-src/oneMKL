.. _onemkl_blas_gemm_ext:

gemm_ext
========

.. container::


   Computes a matrix-matrix product with general matrices.


     **Standard API**

      
     ``gemm_ext`` supports the following precisions and devices.


     .. list-table:: 
        :header-rows: 1

        * -  Ts 
          -  Ta 
          -  Tb 
          -  Tc 
        * -  ``float`` 
          -  ``half`` 
          -  ``half`` 
          -  ``float`` 
        * -  ``half`` 
          -  ``half`` 
          -  ``half`` 
          -  ``half`` 
        * -  ``float`` 
          -  ``float`` 
          -  ``float`` 
          -  ``float`` 
        * -  ``double`` 
          -  ``double`` 
          -  ``double`` 
          -  ``double`` 
        * -  ``std::complex<float>`` 
          -  ``std::complex<float>`` 
          -  ``std::complex<float>`` 
          -  ``std::complex<float>`` 
        * -  ``std::complex<double>`` 
          -  ``std::complex<double>`` 
          -  ``std::complex<double>`` 
          -  ``std::complex<double>`` 


     **Offset API**


     ``gemm_ext`` supports the following precisions.


     .. list-table:: 
        :header-rows: 1

        * -  Ts 
          -  Ta 
          -  Tb 
          -  Tc 
        * -  ``float`` 
          -  ``int8_t`` 
          -  ``uint8_t`` 
          -  ``int32_t`` 


.. container:: section


   .. rubric:: Description
      :class: sectiontitle


   The gemm_ext routines compute a scalar-matrix-matrix product and
   add the result to a scalar-matrix product, with general matrices.
   
   For Standard API, the operation is defined as:

   ::


      C ← alpha*op(A)*op(B) + beta*C 


   For Offset API, the operation is defined as: 

   ::


      C ← alpha*(op(A) - A_offset)*(op(B) - B_offset) + beta*C + C_offset

   where:


   op(X) is one of op(X) = X, or op(X) = X\ :sup:`T`, or op(X) = X\ :sup:`H`


   ``alpha`` and ``beta`` are scalars


   ``A_offset`` is an ``m``-by-``k`` matrix with every element equal to the value ao


   ``B_offset`` is a ``k``-by-``n`` matrix with every element equal to the value bo


   ``C_offset`` is an ``m``-by-``n`` matrix defined by the 
   co buffer as described below. 


   ``A``, ``B``, and ``C`` are matrices


   op(``A``) is ``m`` x ``k``, op(``B``) is ``k`` x ``n``, and
   ``C`` is ``m`` x ``n``.


gemm_ext (Buffer Version)
-------------------------

.. container::

   .. container:: section

      .. rubric:: Syntax
        :class: sectiontitle
      
      
      **Standard API**
      
      
      .. container:: dlsyntaxpara
      
      
        .. cpp:function::  void oneapi::mkl::blas::gemm_ext(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n, std::int64_t k, Ts alpha, sycl::buffer<Ta,1> &a, std::int64_t lda, sycl::buffer<Tb,1> &b, std::int64_t ldb, Ts beta, sycl::buffer<Tc,1> &c, std::int64_t ldc)
      
      
      **Offset API**
      
      
      .. container:: dlsyntaxpara
      
      
        .. cpp:function::  void oneapi::mkl::blas::gemm_ext(sycl::queue &queue, transpose transa, transpose transb, offset offset_type, std::int64_t m, std::int64_t n, std::int64_t k, Ts alpha, sycl::buffer<Ta,1> &a, std::int64_t lda, Ta ao, sycl::buffer<Tb,1> &b, std::int64_t ldb, Tb bo, Ts beta, sycl::buffer<Tc,1> &c, std::int64_t ldc, sycl::buffer<Tc,1> &co)
      
      
   .. container:: section
      
      
      .. rubric:: Input Parameters
         :class: sectiontitle
    
    
      queue
         The queue where the routine should be executed.
    
    
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
    
    
    
      offset_type (offset API only)
         Specifies the form of ``C_offset`` used in the matrix
         multiplication. See
         :ref:`onemkl_datatypes` for
         more details.
    
    
      m
         Number of rows of op(``A``) and ``C``. Must be at least zero.
    
    
      n
         Number of columns of op(``B``) and ``C``. Must be at least
         zero.
    
    
      k
         Number of columns of op(``A``) and rows of op(``B``). Must be
         at least zero.
    
    
      alpha
         Scaling factor for the matrix-matrix product.
    
    
      a
         Buffer holding the input matrix ``A``.
    
    
         If ``A`` is not transposed, ``A`` is an ``m``-by-``k`` matrix
         so the array ``a`` must have size at least ``lda``\ \*\ ``k``.
    
    
         If ``A`` is transposed, ``A`` is a ``k``-by-``m`` matrix so the
         array ``a`` must have size at least ``lda``\ \*\ ``m``.
    
    
         See `Matrix
         Storage <../matrix-storage.html>`__ for
         more details.
    
    
      lda
         Leading dimension of ``A``. Must be at least ``m`` if ``A`` is
         not transposed, and at least ``k`` if ``A`` is transposed. Must
         be positive.
    
    
      ao (offset API only)
         Specifies the scalar offset value for matrix ``A``.
    
    
      b
         Buffer holding the input matrix ``B``.
    
    
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
    
    
      bo (offset API only)
         Specifies the scalar offset value for matrix ``B``.
    
    
      beta
         Scaling factor for matrix ``C``.
    
    
      c
         Buffer holding the input/output matrix ``C``. Must have size at least
         ``ldc`` \* ``n``. See `Matrix
         Storage <../matrix-storage.html>`__ for
         more details.
    
    
      ldc
         Leading dimension of ``C``. Must be positive and at least
         ``m``.
    
    
      co (offset API only)
         Buffer holding the offset values for matrix ``C``.
    
    
         If ``offset_type = offset::fix``, the ``co`` array must have
         size at least 1.
    
    
         If ``offset_type = offset::col``, the ``co`` array must have
         size at least ``max(1,m)``.
    
    
         If ``offset_type = offset::row``, the ``co`` array must have
         size at least ``max(1,n)``. 
    
    
   .. container:: section
    
    
      .. rubric:: Output Parameters
         :class: sectiontitle
    
    
      c
         Output buffer, overwritten by alpha\*op(``A``)*op(``B``) +
         beta\*\ ``C`` for the standard API and alpha\*(op(``A``) -
         ``A_offset``)*(op(``B``) - ``B_offset``) + beta\*\ ``C`` +
         ``C_offset`` for the offset API.
    
    
   .. container:: section
    
    
      .. rubric:: Notes
         :class: sectiontitle
    
    
      If ``beta`` = 0, matrix ``C`` does not need to be initialized
      before calling gemm_ext.



.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-like-extensions`
      


