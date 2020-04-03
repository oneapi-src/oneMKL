.. _hbmv:

hbmv
====


.. container::


   Computes a matrix-vector product using a Hermitian band matrix.


   .. container:: section
      :name: GUID-F5FF420B-922B-4552-8F55-6EBCA7177881


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void hbmv(queue &exec_queue, uplo upper_lower,      std::int64_t n, std::int64_t k, T alpha, buffer<T,1> &a,      std::int64_t lda, buffer<T,1> &x, std::int64_t incx, T beta,      buffer<T,1> &y, std::int64_t incy)

      ``hbmv`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 




.. container:: section
   :name: GUID-8AB4BAC9-8124-4B52-8C15-1BC673820EB9


   .. rubric:: Description
      :name: description
      :class: sectiontitle


   The hbmv routines compute a scalar-matrix-vector product and add the
   result to a scalar-vector product, with a Hermitian band matrix. The
   operation is defined as


  


      y <- alpha*A*x + beta*y


   where:


   ``alpha`` and ``beta`` are scalars,


   ``A`` is an ``n``-by-``n`` Hermitian band matrix, with ``k``
   super-diagonals,


   ``x`` and ``y`` are vectors of length ``n``.


.. container:: section
   :name: GUID-E1436726-01FE-4206-871E-B905F59A96B4


   .. rubric:: Input Parameters
      :name: input-parameters
      :class: sectiontitle


   exec_queue
      The queue where the routine should be executed.


   upper_lower
      Specifies whether ``A`` is upper or lower triangular. See
      :ref:`onemkl_datatypes` for more
      details.


   n
      Number of rows and columns of ``A``. Must be at least zero.


   k
      Number of super-diagonals of the matrix ``A``. Must be at least
      zero.


   alpha
      Scaling factor for the matrix-vector product.


   a
      Buffer holding input matrix ``A``. Must have size at least
      ``lda``\ \*\ ``n``. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   lda
      Leading dimension of matrix ``A``. Must be at least (``k`` + 1),
      and positive.


   x
      Buffer holding input vector ``x``. The buffer must be of size at
      least (1 + (``m`` - 1)*abs(``incx``)). See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   incx
      Stride of vector ``x``.


   beta
      Scaling factor for vector ``y``.


   y
      Buffer holding input/output vector ``y``. The buffer must be of
      size at least (1 + (``n`` - 1)*abs(``incy``)). See `Matrix and
      Vector Storage <../matrix-storage.html>`__
      for more details.


   incy
      Stride of vector ``y``.


.. container:: section
   :name: GUID-7261182A-450B-46F5-8C61-7133597D3530


   .. rubric:: Output Parameters
      :name: output-parameters
      :class: sectiontitle


   y
      Buffer holding the updated vector ``y``.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-2-routines`
      


.. container::

