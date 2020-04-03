.. _her:

her
===


.. container::


   Computes a rank-1 update of a Hermitian matrix.


   .. container:: section
      :name: GUID-252B1D4A-30C7-4678-9793-6A0C90DEB04A


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void her(queue &exec_queue, uplo upper_lower,      std::int64_t n, T alpha, buffer<T,1> &x, std::int64_t incx,      buffer<T,1> &a, std::int64_t lda)

      ``her`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 




.. container:: section
   :name: GUID-A06B7C00-CFD6-4A01-9739-19093823B58E


   .. rubric:: Description
      :name: description
      :class: sectiontitle


   The her routines compute a scalar-vector-vector product and add the
   result to a Hermitian matrix. The operation is defined as


  


      A <- alpha*x*x :sup:`H` + A


   where:


   ``alpha`` is scalar,


   ``A`` is an ``n``-by-``n`` Hermitian matrix,


   ``x`` is a vector of length ``n``.


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


   x
      Buffer holding input vector ``x``. The buffer must be of size at
      least (1 + (``n`` - 1)*abs(``incx``)). See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   incx
      Stride of vector ``x``.


   a
      Buffer holding input matrix ``A``. Must have size at least
      ``lda``\ \*\ ``n``. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   lda
      Leading dimension of matrix ``A``. Must be at least ``n``, and
      positive.


.. container:: section
   :name: GUID-89A60481-0763-4608-B346-3CC746467F28


   .. rubric:: Output Parameters
      :name: output-parameters
      :class: sectiontitle


   a
      Buffer holding the updated upper triangular part of theHermitian
      matrix ``A`` if ``upper_lower = upper`` or the updated
      lowertriangular part of the Hermitian matrix ``A`` if
      ``upper_lower = lower``.


      The imaginary parts of the diagonal elementsare set to zero.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-2-routines`
      


.. container::

