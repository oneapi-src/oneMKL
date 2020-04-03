.. _symv:

symv
====


.. container::


   Computes a matrix-vector product for a symmetric matrix.


   .. container:: section
      :name: GUID-1E9C9EA9-0366-420E-A704-AB605C8ED92A


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void symv(queue &exec_queue, uplo upper_lower,      std::int64_t n, T alpha, buffer<T,1> &a, std::int64_t lda,      buffer<T,1> &x, std::int64_t incx, T beta, buffer<T,1> &y,      std::int64_t incy)

      ``symv`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``float`` 
         * -  ``double`` 




.. container:: section
   :name: GUID-DE8D8321-D53D-4226-A940-CDE0E720EC95


   .. rubric:: Description
      :name: description
      :class: sectiontitle


   The symv routines routines compute a scalar-matrix-vector product and
   add the result to a scalar-vector product, with a symmetric matrix.
   The operation is defined as


  


      y <- alpha*A*x + beta*y


   where:


   ``alpha`` and ``beta`` are scalars,


   ``A`` is an ``n``-by-``n`` symmetric matrix,


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


   y
      Buffer holding input/output vector ``y``. The buffer must be of
      size at least (1 + (``n`` - 1)*abs(``incy``)). See `Matrix and
      Vector Storage <../matrix-storage.html>`__
      for more details.


   incy
      Stride of vector ``y``.


.. container:: section
   :name: GUID-E16C8443-A2A4-483C-9D46-FF428E80FEB0


   .. rubric:: Output Parameters
      :name: output-parameters
      :class: sectiontitle


   y
      Buffer holding the updated vector ``y``.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-2-routines`
      


.. container::

