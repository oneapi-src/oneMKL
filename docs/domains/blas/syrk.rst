.. _syrk:

syrk
====


.. container::


   Performs a symmetric rank-k update.


   .. container:: section
      :name: GUID-F8123F9B-A182-4BDB-A1A3-90FEC4F56231


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void syrk(queue &exec_queue, uplo upper_lower,      transpose trans, std::int64_t n, std::int64_t k, T alpha,      buffer<T,1> &a, std::int64_t lda, T beta, buffer<T,1> &c,      std::int64_t ldc)

      syrk supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``float`` 
         * -  ``double`` 
         * -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 




.. container:: section
   :name: GUID-8E133139-EE58-44B8-A507-2263BDD1399B


   .. rubric:: Description
      :name: description
      :class: sectiontitle


   The syrk routines perform a rank-k update of a symmetric matrix ``C``
   by a general matrix ``A``. The operation is defined as:


  


      C <- alpha*op(A)*op(A)T + beta*C


   where:


   op(``X``) is one of op(``X``) = ``X`` or op(``X``) = ``X``\ :sup:`T`
   ,


   ``alpha`` and ``beta`` are scalars,


   ``C`` is a symmetric matrix and ``A``\ is a general matrix.


   Here op(``A``) is ``n``-by-``k``, and ``C`` is ``n``-by-``n``.


.. container:: section
   :name: GUID-96D007CC-23F0-46FA-9085-6DBFC5BB30E6


   .. rubric:: Input Parameters
      :name: input-parameters
      :class: sectiontitle


   exec_queue
      The queue where the routine should be executed.


   upper_lower
      Specifies whether ``A``'s data is stored in its upper or lower
      triangle. See
      :ref:`onemkl_datatypes` for more
      details.


   trans
      Specifies op(``A``), the transposition operation applied to ``A``
      (See
      :ref:`onemkl_datatypes` for more
      details). Conjugation is never performed, even if ``trans`` =
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
   :name: GUID-E14CE68E-2E28-48BB-8FD7-B84A21563BDA


   .. rubric:: Output Parameters
      :name: output-parameters
      :class: sectiontitle


   c
      Output buffer, overwritten by
      ``alpha``\ \*op(``A``)*op(``A``)\ :sup:`T` + ``beta``\ \*\ ``C``.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-3-routines`
      


.. container::

