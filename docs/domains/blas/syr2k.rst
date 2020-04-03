.. _syr2k:

syr2k
=====


.. container::


   Performs a symmetric rank-2k update.


   .. container:: section
      :name: GUID-EED2648B-6435-4DD1-AC36-21039DFC61DD


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void syr2k(queue &exec_queue, uplo upper_lower,      transpose trans, std::int64_t n, std::int64_t k, T alpha,      buffer<T,1> &a, std::int64_t lda, buffer<T,1> &b, std::int64_t      ldb, T beta, buffer<T,1> &c, std::int64_t ldc)

      syr2k supports the following precisions:


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``float`` 
         * -  ``double`` 
         * -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 




.. container:: section
   :name: GUID-1FB46B8F-1B13-4A6B-A3A5-0A5B34049068


   .. rubric:: Description
      :name: description
      :class: sectiontitle


   The syr2k routines perform a rank-2k update of an ``n`` x ``n``
   symmetric matrix ``C`` by general matrices ``A`` and ``B``. If
   ``trans`` = ``transpose::nontrans``, the operation is defined as:


  


      C <- alpha*(A*B :sup:`T` + B*A :sup:`T`) + beta*C


   where ``A`` is ``n`` x ``k`` and ``B`` is ``k`` x ``n``.


   If ``trans`` = ``transpose::trans``, the operationis defined as:


  


      C <- alpha*(A :sup:`T`*B + B :sup:`T`*A) + beta*C


   where ``A`` is ``k`` x ``n`` and ``B`` is ``n`` x ``k``.


   In both cases:


   ``alpha`` and ``beta`` are scalars,


   ``C`` is a symmetric matrix and ``A``,\ ``B`` are general matrices,


   The inner dimension of both matrix multiplications is ``k``.


.. container:: section
   :name: GUID-3EBEFBDD-93AF-4376-9BA2-A7042179BF13


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
   :name: GUID-5779F783-54BC-4887-9CBB-96B8EC9F00E9


   .. rubric:: Output Parameters
      :name: output-parameters
      :class: sectiontitle


   c
      Output buffer, overwritten by the updated C matrix.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-3-routines`
      


.. container::

