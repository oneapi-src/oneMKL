.. _syr2:

syr2
====


.. container::


   Computes a rank-2 update of a symmetric matrix.


   .. container:: section
      :name: GUID-580F2222-D47E-43A3-B9A2-037F353825D5


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void syr2(queue &exec_queue, uplo upper_lower,      std::int64_t n, T alpha, buffer<T,1> &x, std::int64_t incx,      buffer<T,1> &y, std::int64_t incy, buffer<T,1> &a, std::int64_t      lda)

      ``syr2`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``float`` 
         * -  ``double`` 




.. container:: section
   :name: GUID-CDA05459-F2FE-4933-A552-D6E52EC46D13


   .. rubric:: Description
      :name: description
      :class: sectiontitle


   The syr2 routines compute two scalar-vector-vector product add them
   and add the result to a matrix, with a symmetric matrix. The
   operation is defined as


  


      A <- alpha*x*y :sup:`T` + alpha*y*x :sup:`T` + A


   where:


   ``alpha`` is a scalar,


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
      Number of columns of ``A``. Must be at least zero.


   alpha
      Scaling factor for the matrix-vector product.


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


   a
      Buffer holding input matrix ``A``. Must have size at least
      ``lda``\ \*\ ``n``. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   lda
      Leading dimension of matrix ``A``. Must be at least ``n``, and
      positive.


.. container:: section
   :name: GUID-6992A39F-8AB7-42D9-B126-4F8ECF9C1ECE


   .. rubric:: Output Parameters
      :name: output-parameters
      :class: sectiontitle


   a
      Buffer holding the updated upper triangularpart of the symmetric
      matrix ``A`` if ``upper_lower =upper``, or the updated lower
      triangular part of thesymmetric matrix ``A`` if
      ``upper_lower =lower``.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-2-routines`
      


.. container::

