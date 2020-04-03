.. _hemv:

hemv
====


.. container::


   Computes a matrix-vector product using a Hermitian matrix.


   .. container:: section
      :name: GUID-152B72DC-F67F-4D7D-96DA-67AE6AD41718


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void hemv(queue &exec_queue, uplo upper_lower,      std::int64_t n, T alpha, buffer<T,1> &a, std::int64_t lda,      buffer<T,1> &x, std::int64_t incx, T beta, buffer<T,1> &y,      std::int64_t incy)

      ``hemv`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 




.. container:: section
   :name: GUID-0E4AE01A-4FE8-42AC-B236-409F4DD48F88


   .. rubric:: Description
      :name: description
      :class: sectiontitle


   The hemv routines compute a scalar-matrix-vector product and add the
   result to a scalar-vector product, with a Hermitian matrix. The
   operation is defined as


  


      y <- alpha*A*x + beta*y


   where:


   ``alpha`` and ``beta`` are scalars,


   ``A`` is an ``n``-by-``n`` Hermitian matrix,


   ``x`` and ``y`` are vectors of length ``n``.


.. container:: section
   :name: GUID-E1436726-01FE-4206-871E-B905F59A96B4


   .. rubric:: Input Parameters
      :name: input-parameters
      :class: sectiontitle


   exec_queue
      The queue where the routine should be executed.


   upper_lower
      Specifies whether *A* is upper or lower triangular. See
      :ref:`onemkl_datatypes` for more
      details.


   n
      Number of rows and columns of ``A``. Must be at least zero.


   alpha
      Scaling factor for the matrix-vector product.


   a
      Buffer holding input matrix ``A``. Must have size at least
      ``lda``\ \*\ ``n``. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   lda
      Leading dimension of matrix ``A``. Must be at least ``m``, and
      positive.


   x
      Buffer holding input vector ``x``. The buffer must be of size at
      least (1 + (``n`` - 1)*abs(``incx``)). See `Matrix and Vector
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
   :name: GUID-66566E59-9A52-4207-B123-AF45FA3A0FBC


   .. rubric:: Output Parameters
      :name: output-parameters
      :class: sectiontitle


   y
      Buffer holding the updated vector ``y``.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-2-routines`
      


.. container::

