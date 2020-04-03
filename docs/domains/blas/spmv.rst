.. _spmv:

spmv
====


.. container::


   Computes a matrix-vector product with a symmetric packed matrix.


   .. container:: section
      :name: GUID-BCC82B03-92EB-4D73-B69C-8AE8646FBEAC


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void spmv(queue &exec_queue, uplo upper_lower,      std::int64_t n, T alpha, buffer<T,1> &a, buffer<T,1> &x,      std::int64_t incx, T beta, buffer<T,1> &y, std::int64_t incy)

      ``spmv`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``float`` 
         * -  ``double`` 




.. container:: section
   :name: GUID-D27BBFFF-79F4-4236-96A6-B305FA1858B0


   .. rubric:: Description
      :name: description
      :class: sectiontitle


   The spmv routines compute a scalar-matrix-vector product and add the
   result to a scalar-vector product, with a symmetric packed matrix.
   The operation is defined as


  


      y <- alpha*A*x + beta*y


   where:


   ``alpha`` and ``beta`` are scalars,


   ``A`` is an ``n``-by-``n`` symmetric matrix, supplied in packed form.


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


   alpha
      Scaling factor for the matrix-vector product.


   a
      Buffer holding input matrix ``A``. Must have size at least
      (``n``\ \*(``n``\ +1))/2. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


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
   :name: GUID-23FF1F5C-5560-40B6-807D-B6352FA320D6


   .. rubric:: Output Parameters
      :name: output-parameters
      :class: sectiontitle


   y
      Buffer holding the updated vector ``y``.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-2-routines`
      


.. container::

