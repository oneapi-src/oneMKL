.. _onemkl_blas_syrk:

syrk
====


.. container::


   Performs a symmetric rank-k update.



      ``syrk`` supports the following precisions.


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


   The ``syrk`` routines perform a rank-k update of a symmetric matrix ``C``
   by a general matrix ``A``. The operation is defined as:



      C <- alpha*op(A)*op(A)T + beta*C


   where:


   op(``X``) is one of op(``X``) = ``X`` or op(``X``) = ``X``\ :sup:`T`
   ,


   ``alpha`` and ``beta`` are scalars,


   ``C`` is a symmetric matrix and ``A``\ is a general matrix.


   Here op(``A``) is ``n``-by-``k``, and ``C`` is ``n``-by-``n``.


syrk (Buffer Version)
---------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. cpp:function::  void oneapi::mkl::blas::syrk(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k, T alpha, sycl::buffer<T,1> &a, std::int64_t lda, T beta, sycl::buffer<T,1> &c, std::int64_t ldc)

.. container:: section


   .. rubric:: Input Parameters
      :class: sectiontitle


   queue
      The queue where the routine should be executed.


   upper_lower
      Specifies whether ``A``'s data is stored in its upper or lower
      triangle. See :ref:`onemkl_datatypes` for more details.


   trans
      Specifies op(``A``), the transposition operation applied to ``A`` (See :ref:`onemkl_datatypes` for more details). Conjugation is never performed, even if ``trans`` =
      ``transpose::conjtrans``.


   n
      Number of rows and columns in ``C``. The value of ``n`` must be at
      least zero.


   k
      Number of columns in op(``A``).The value of ``k`` must be at least
      zero.


   alpha
      Scaling factor for the rank-``k`` update.


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
      Scaling factor for matrix ``C``.


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
      Output buffer, overwritten by
      ``alpha``\ \*op(``A``)*op(``A``)\ :sup:`T` + ``beta``\ \*\ ``C``.


syrk (USM Version)
------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. container:: dlsyntaxpara


         .. cpp:function::  sycl::event oneapi::mkl::blas::syrk(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k, T alpha, const T* a, std::int64_t lda, T beta, T* c, std::int64_t ldc, const sycl::vector_class<sycl::event> &dependencies = {})
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
         ``A`` (See :ref:`onemkl_datatypes` for more details). Conjugation is never performed, even if
         ``trans`` = ``transpose::conjtrans``.


      n
         Number of rows and columns in ``C``. The value of ``n`` must be
         at least zero.


      k
         Number of columns in op(``A``). The value of ``k`` must be at
         least zero.


      alpha
         Scaling factor for the rank-``k`` update.


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
         Scaling factor for matrix ``C``.


      c
         Pointer to input/output matrix ``C``. Must have size at least
         ``ldc``\ \*\ ``n``. See `Matrix and Vector
         Storage <../matrix-storage.html>`__ for
         more details.


      ldc
         Leading dimension of ``C``. Must be positive and at least
         ``n``.


   .. container:: section


      .. rubric:: Output Parameters
         :class: sectiontitle


      c
         Pointer to the output matrix, overwritten by
         ``alpha``\ \*op(``A``)*op(``A``)\ :sup:`T` +
         ``beta``\ \*\ ``C``.


   .. container:: section


      .. rubric:: Return Values
         :class: sectiontitle


      Output event to wait on to ensure computation is complete.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-3-routines`
