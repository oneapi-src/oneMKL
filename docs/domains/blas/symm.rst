.. _symm:

symm
====


.. container::


   Computes a matrix-matrix product where one input matrix is symmetric
   and one matrix is general.


   .. container:: section
      :name: GUID-BFE36A6B-941E-4B49-AB0E-CFB687B1AD64


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void symm(queue &exec_queue, side left_right,      uplo upper_lower, std::int64_t m, std::int64_t n, T alpha,      buffer<T,1> &a, std::int64_t lda, buffer<T,1> &b, std::int64_t      ldb, T beta, buffer<T,1> &c, std::int64_t ldc)

      symm supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``float`` 
         * -  ``double`` 
         * -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 




.. container:: section
   :name: GUID-E8FE37B0-C527-4AA6-B57F-AE3F4843F23A


   .. rubric:: Description
      :name: description
      :class: sectiontitle


   The symm routines compute a scalar-matrix-matrix product and add the
   result to a scalar-matrix product, where one of the matrices in the
   multiplication is symmetric. The argument ``left_right`` determines
   if the symmetric matrix, ``A``, is on the left of the multiplication
   (``left_right`` = ``side::left``) or on the right (``left_right`` =
   ``side::right``). Depending on ``left_right``, the operation is
   defined as


  


      C <- alpha*A*B + beta*C,


   or


  


      C <- alpha*B*A + beta*C,


   where:


   ``alpha`` and ``beta`` are scalars,


   ``A`` is a symmetric matrix, either ``m``-by-``m`` or ``n``-by-``n``,


   ``B`` and ``C`` are ``m``-by-``n`` matrices.


.. container:: section
   :name: GUID-70716375-C54E-4AA6-94DC-65AF79D46BB2


   .. rubric:: Input Parameters
      :name: input-parameters
      :class: sectiontitle


   exec_queue
      The queue where the routine should be executed.


   left_right
      Specifies whether ``A`` is on the left side of the multiplication
      (``side::left``) or on the right side (``side::right``). See
      :ref:`onemkl_datatypes` for more
      details.


   upper_lower
      Specifies whether *A*'s data is stored in its upper or lower
      triangle. See
      :ref:`onemkl_datatypes` for more
      details.


   m
      Number of rows of ``B`` and ``C``. The value of ``m`` must be at
      least zero.


   n
      Number of columns of ``B`` and ``C``. The value of ``n`` must be
      at least zero.


   alpha
      Scaling factor for the matrix-matrix product.


   a
      Buffer holding input matrix ``A``. Must have size at least
      ``lda``\ \*\ ``m`` if ``A`` is on the left of the multiplication,
      or ``lda``\ \*\ ``n`` if ``A`` is on the right. See `Matrix and
      Vector Storage <../matrix-storage.html>`__
      for more details.


   lda
      Leading dimension of ``A``. Must be at least ``m`` if ``A`` is on
      the left of the multiplication, or at least ``n`` if ``A`` is on
      the right. Must be positive.


   b
      Buffer holding input matrix ``B``. Must have size at least
      ``ldb``\ \*\ ``n``. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   ldb
      Leading dimension of ``B``. Must be positive and at least ``m``.


   beta
      Scaling factor for matrix ``C``.


   c
      Buffer holding input/output matrix ``C``. Must have size at least
      ``ldc``\ \*\ ``n``. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   ldc
      Leading dimension of ``C``. Must be positive and at least ``m``.


.. container:: section
   :name: GUID-DD569858-5D3C-4565-8BAB-FE548427DCF2


   .. rubric:: Output Parameters
      :name: output-parameters
      :class: sectiontitle


   c
      Output buffer, overwritten by ``alpha``\ \*\ ``A``\ \*\ ``B`` +
      ``beta``\ \*\ ``C`` (``left_right`` = ``side::left``) or
      ``alpha``\ \*\ ``B``\ \*\ ``A`` + ``beta``\ \*\ ``C``
      (``left_right`` = ``side::right``).


.. container:: section
   :name: EXAMPLE_5EF48B8A07D849EA84A74FE22F0D5B24


   .. rubric:: Notes
      :name: notes
      :class: sectiontitle


   If ``beta`` = 0, matrix ``C`` does not need to be initialized before
   calling ``symm``.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-3-routines`
      


.. container::

