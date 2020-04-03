.. _herk:

herk
====


.. container::


   Performs a Hermitian rank-k update.


   .. container:: section
      :name: GUID-407B8203-A28D-468B-BA79-87FA865E75A2


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void herk(queue &exec_queue, uplo upper_lower,      transpose trans, std::int64_t n, std::int64_t k, T_real alpha,      buffer<T,1> &a, std::int64_t lda, T_real beta, buffer<T,1> &c,      std::int64_t ldc)

      herk supports the following precisions:


      .. list-table:: 
         :header-rows: 1

         * -  T 
           -  T_real 
         * -  ``std::complex<float>`` 
           -  ``float`` 
         * -  ``std::complex<double>`` 
           -  ``double`` 




.. container:: section
   :name: GUID-539B4E63-9CDF-4834-999A-4133CE5DE1E5


   .. rubric:: Description
      :name: description
      :class: sectiontitle


   The herk routines compute a rank-``k`` update of a Hermitian matrix
   *C* by a general matrix ``A``. The operation is defined as:


  


      C <- alpha*op(A)*op(A) :sup:`H` + beta*C


   where:


   op(``X``) is one of op(``X``) = ``X`` or op(``X``) = ``X``\ :sup:`H`,


   ``alpha`` and ``beta`` are real scalars,


   ``C`` is a Hermitian matrix and ``A`` is a general matrix.


   Here op(``A``) is ``n`` x ``k``, and ``C`` is ``n`` x ``n``.


.. container:: section
   :name: GUID-7B880A06-4E53-4DE9-B0E6-D70673CF2638


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
      Specifies op(``A``), the transposition operation applied to ``A``.
      See
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
   :name: GUID-05309970-DEC8-4D87-90AA-958FC101E119


   .. rubric:: Output Parameters
      :name: output-parameters
      :class: sectiontitle


   c
      The output buffer, overwritten by
      ``alpha``\ \*op(``A``)*op(``A``)\ :sup:`T` + ``beta``\ \*\ ``C``.
      The imaginary parts of the diagonal elements are set to zero.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-3-routines`
      


.. container::

