.. _hpmv:

hpmv
====


.. container::


   Computes a matrix-vector product using a Hermitian packed matrix.


   .. container:: section
      :name: GUID-C6E4A4A7-5CBE-46ED-A021-8FEAABAA2E93


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void hpmv(queue &exec_queue, uplo upper_lower,      std::int64_t n, T alpha, buffer<T,1> &a, buffer<T,1> &x,      std::int64_t incx, T beta, buffer<T,1> &y, std::int64_t incy)

      ``hpmv`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 




.. container:: section
   :name: GUID-A95C32C5-0371-429B-847C-4EE29FD9C480


   .. rubric:: Description
      :name: description
      :class: sectiontitle


   The hpmv routines compute a scalar-matrix-vector product and add the
   result to a scalar-vector product, with a Hermitian packed matrix.
   The operation is defined as


  


      y <- alpha*A*x + beta*y


   where:


   ``alpha`` and ``beta`` are scalars,


   ``A`` is an ``n``-by-``n`` Hermitian matrix supplied in packed form,


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
      (``n``\ \*(``n``\ +1))/2. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


      The imaginary parts of the diagonal elements need not be set and
      are assumed to be zero.


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
   :name: GUID-416B82CD-C5B8-472A-8347-04997EA6D6E6


   .. rubric:: Output Parameters
      :name: output-parameters
      :class: sectiontitle


   y
      Buffer holding the updated vector ``y``.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-2-routines`
      


.. container::

