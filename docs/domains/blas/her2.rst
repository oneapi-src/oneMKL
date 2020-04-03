.. _her2:

her2
====


.. container::


   Computes a rank-2 update of a Hermitian matrix.


   .. container:: section
      :name: GUID-4BED3537-E900-4260-A6EB-2F42CB1D3AFB


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void her2(queue &exec_queue, uplo upper_lower,      std::int64_t n, T alpha, buffer<T,1> &x, std::int64_t incx,      buffer<T,1> &y, std::int64_t incy, buffer<T,1> &a, std::int64_t      lda)

      ``her2`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 




.. container:: section
   :name: GUID-2B939041-9BCC-4AE8-A31D-2CFCA67B9B6A


   .. rubric:: Description
      :name: description
      :class: sectiontitle


   The her2 routines compute two scalar-vector-vector products and add
   them to a Hermitian matrix. The operation is defined as


  


      A <- alpha*x*y :sup:`H` + conjg(alpha)*y*x :sup:`H` + A


   where:


   ``alpha`` is a scalar,


   ``A`` is an ``n``-by-``n`` Hermitian matrix.


   ``x`` and ``y`` are vectors or length ``n``.


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
   :name: GUID-34B3837B-4980-458B-AC3A-EEE5F635834C


   .. rubric:: Output Parameters
      :name: output-parameters
      :class: sectiontitle


   a
      Buffer holding the updated upper triangular part of theHermitian
      matrix ``A`` if ``upper_lower = upper``, or the updated
      lowertriangular part of the Hermitian matrix ``A`` if
      ``upper_lower = lower``.


      The imaginary parts of the diagonal elementsare set to zero.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-2-routines`
      


.. container::

