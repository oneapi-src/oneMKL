.. _ger:

ger
===


.. container::


   Computes a rank-1 update of a general matrix.


   .. container:: section
      :name: GUID-0DA23698-EB19-4AAF-A5FD-9BB530A9EFE0


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void ger(queue &exec_queue, std::int64_t m,      std::int64_t n, T alpha, buffer<T,1> &x, std::int64_t incx,      buffer<T,1> &y, std::int64_t incy, buffer<T,1> &a, std::int64_t      lda)

      ``ger`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``float`` 
         * -  ``double`` 




.. container:: section
   :name: GUID-72E035B0-E1C2-442B-AE9D-2CB873E90FAF


   .. rubric:: Description
      :name: description
      :class: sectiontitle


   The ger routines compute a scalar-vector-vector product and add the
   result to a general matrix. The operation is defined as


  


      A <- alpha*x*y :sup:`T` + A


   where:


   ``alpha`` is scalar,


   ``A`` is an ``m``-by-``n`` matrix,


   ``x`` is a vector length ``m``,


   ``y`` is a vector length ``n``.


.. container:: section
   :name: GUID-6953A2E5-0065-425C-986B-15966C793067


   .. rubric:: Input Parameters
      :name: input-parameters
      :class: sectiontitle


   exec_queue
      The queue where the routine should be executed.


   m
      Number of rows of ``A``. Must be at least zero.


   n
      Number of columns of ``A``. Must be at least zero.


   alpha
      Scaling factor for the matrix-vector product.


   x
      Buffer holding input vector ``x``. The buffer must be of size at
      least (1 + (``m`` - 1)*abs(``incx``)). See `Matrix and Vector
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


   a
      Buffer holding input matrix ``A``. Must have size at least
      ``lda``\ \*\ ``n``. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   lda
      Leading dimension of matrix ``A``. Must be at least ``m``, and
      positive.


.. container:: section
   :name: GUID-E2A13688-1D12-4DD0-9752-3557E980ACC0


   .. rubric:: Output Parameters
      :name: output-parameters
      :class: sectiontitle


   a
      Buffer holding the updated matrix ``A``.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-2-routines`
      


.. container::

