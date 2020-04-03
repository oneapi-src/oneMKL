.. _gerc:

gerc
====


.. container::


   Computes a rank-1 update (conjugated) of a general complex matrix.


   .. container:: section
      :name: GUID-5A1B0292-28F6-45EB-95C4-FDA03D8D5062


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void gerc(queue &exec_queue, std::int64_t m,      std::int64_t n, T alpha, buffer<T,1> &x, std::int64_t incx,      buffer<T,1> &y, std::int64_t incy, buffer<T,1> &a, std::int64_t      lda)

      ``gerc`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 




.. container:: section
   :name: GUID-6CB627E5-A9C7-488D-8366-E7944A5C889E


   .. rubric:: Description
      :name: description
      :class: sectiontitle


   The gerc routines compute a scalar-vector-vector product and add the
   result to a general matrix. The operation is defined as


  


      A <- alpha*x*y :sup:`H` + A


   where:


   ``alpha`` is a scalar,


   ``A`` is an ``m``-by-``n`` matrix,


   ``x`` is a vector of length ``m``,


   ``y`` is vector of length ``n``.


.. container:: section
   :name: GUID-E1436726-01FE-4206-871E-B905F59A96B4


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
   :name: GUID-48944ED2-C10F-4B64-A91A-C9050AD24A92


   .. rubric:: Output Parameters
      :name: output-parameters
      :class: sectiontitle


   a
      Buffer holding the updated matrix *A*.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-2-routines`
      


.. container::

