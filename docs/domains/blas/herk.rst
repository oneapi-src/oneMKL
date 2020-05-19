.. _onemkl_blas_herk:

herk
====


.. container::


   Performs a Hermitian rank-k update.



      ``herk`` supports the following precisions:


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


   The ``herk`` routines compute a rank-``k`` update of a Hermitian matrix
   ``C`` by a general matrix ``A``. The operation is defined as:


      C <- alpha*op(A)*op(A) :sup:`H` + beta*C


   where:


   op(``X``) is one of op(``X``) = ``X`` or op(``X``) = ``X``\ :sup:`H`,


   ``alpha`` and ``beta`` are real scalars,


   ``C`` is a Hermitian matrix and ``A`` is a general matrix.


   Here op(``A``) is ``n`` x ``k``, and ``C`` is ``n`` x ``n``.


herk (Buffer Version)
---------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. cpp:function::  void onemkl::blas::herk(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k, T_real alpha, sycl::buffer<T,1> &a, std::int64_t lda, T_real beta, sycl::buffer<T,1> &c, std::int64_t ldc)

.. container:: section


   .. rubric:: Input Parameters
      :class: sectiontitle


   queue
      The queue where the routine should be executed.


   upper_lower
      Specifies whether ``A``'s data is stored in its upper or lower
      triangle. See :ref:`onemkl_datatypes` for more details.


   trans
      Specifies op(``A``), the transposition operation applied to ``A``. See
      :ref:`onemkl_datatypes` for more
      details. Supported operations are ``transpose::nontrans`` and
      ``transpose::conjtrans``.


   n
      The number of rows and columns in ``C``.The value of ``n`` must be
      at least zero.


   k
      Number of columns in op(``A``).


      The value of ``k`` must be at least zero.


   alpha
      Real scaling factor for the rank-``k`` update.


   a
      Buffer holding input matrix ``A``. If ``trans`` =
      ``transpose::nontrans``, ``A`` is an ``n``-by-``k`` matrix so the
      array ``a`` must have size at least ``lda``\ \*\ ``k``. Otherwise,
      ``A`` is an ``k``-by-``n`` matrix so the array ``a`` must have
      size at least ``lda``\ \*\ ``n``. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   lda
      Leading dimension of ``A``. Must be at least ``n`` if ``A`` is not
      transposed, and at least ``k`` if ``A`` is transposed. Must be
      positive.


   beta
      Real scaling factor for matrix ``C``.


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
      The output buffer, overwritten by
      ``alpha``\ \*op(``A``)*op(``A``)\ :sup:`T` + ``beta``\ \*\ ``C``.
      The imaginary parts of the diagonal elements are set to zero.


herk (USM Version)
------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. container:: dlsyntaxpara


         .. cpp:function::  sycl::event onemkl::blas::herk(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k, T_real alpha, const T* a, std::int64_t lda, T_real beta, T* c, std::int64_t ldc, const sycl::vector_class<sycl::event> &dependencies = {})
   .. container:: section


      .. rubric:: Input Parameters
         :class: sectiontitle


      queue
         The queue where the routine should be executed.


      upper_lower
         Specifies whether ``A``'s data is stored in its upper or lower
         triangle. See :ref:`onemkl_datatypes` for more details.



      trans
         Specifies op(``A``), the transposition operation applied to
         ``A``. See :ref:`onemkl_datatypes` for more details. Supported operations are ``transpose::nontrans``
         and ``transpose::conjtrans``.


      n
         The number of rows and columns in ``C``.The value of ``n`` must
         be at least zero.


      k
         Number of columns in op(``A``).


         The value of ``k`` must be at least zero.


      alpha
         Real scaling factor for the rank-``k`` update.


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
         Leading dimension of ``A``. Must be at least ``n`` if ``A`` is
         not transposed, and at least ``k`` if ``A`` is transposed. Must
         be positive.


      beta
         Real scaling factor for matrix ``C``.


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
         Pointer to the output matrix, overwritten by
         ``alpha``\ \*op(``A``)*op(``A``)\ :sup:`T` +
         ``beta``\ \*\ ``C``. The imaginary parts of the diagonal
         elements are set to zero.


   .. container:: section


      .. rubric:: Return Values
         :class: sectiontitle


      Output event to wait on to ensure computation is complete.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-3-routines`
