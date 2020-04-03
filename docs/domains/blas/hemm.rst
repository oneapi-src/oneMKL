.. _hemm:

hemm
====


.. container::


   Computes a matrix-matrix product where one input matrix is Hermitian
   and one is general.


   .. container:: section
      :name: GUID-F06C86BA-4F57-4608-B0D7-F7B920F867D7


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void hemm(queue &exec_queue, side left_right,      uplo upper_lower, std::int64_t m, std::int64_t n, T alpha,      buffer<T,1> &a, std::int64_t lda, buffer<T,1> &b, std::int64_t      ldb, T beta, buffer<T,1> &c, std::int64_t ldc)

      hemm supports the following precisions:


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 




.. container:: section
   :name: GUID-835E7F58-406E-444F-9DFD-121B84C22284


   .. rubric:: Description
      :name: description
      :class: sectiontitle


   The hemm routines compute a scalar-matrix-matrix product and add the
   result to a scalar-matrix product, where one of the matrices in the
   multiplication is Hermitian. The argument ``left_right`` determines
   if the Hermitian matrix, ``A``, is on the left of the multiplication
   (``left_right`` = ``side::left``) or on the right (``left_right`` =
   ``side::right``). Depending on ``left_right``, the operation is
   defined as


  


      C <- alpha*A*B + beta*C


   or


  


      C <- alpha*B*A + beta*C


   where:


   ``alpha`` and ``beta`` are scalars,


   ``A`` is a Hermitian matrix, either ``m``-by-``m`` or ``n``-by-``n``
   matrices,


   ``B`` and ``C`` are ``m``-by-``n`` matrices.


.. container:: section
   :name: GUID-922C5F92-38B2-457B-B6C7-3CDD0531F97D


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


   uplo
      Specifies whether ``A``'s data is stored in its upper or lower
      triangle. See
      :ref:`onemkl_datatypes` for more
      details.


   m
      Specifies the number of rows of the matrix ``B`` and ``C``.


      The value of ``m`` must be at least zero.


   n
      Specifies the number of columns of the matrix ``B`` and ``C``.


      The value of ``n`` must be at least zero.


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
   :name: GUID-94385C78-968D-4C03-AA5C-7379D5607800


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
   calling ``hemm``.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-3-routines`
      


.. container::

