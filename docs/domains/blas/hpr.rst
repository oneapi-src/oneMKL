.. _onemkl_blas_hpr:

hpr
===


.. container::


   Computes a rank-1 update of a Hermitian packed matrix.



      ``hpr`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 


.. container:: section


   .. rubric:: Description
      :class: sectiontitle


   The ``hpr`` routines compute a scalar-vector-vector product and add the
   result to a Hermitian packed matrix. The operation is defined as


      A <- alpha*x*x :sup:`H` + A


   where:


   ``alpha`` is scalar,


   ``A`` is an ``n``-by-``n`` Hermitian matrix, supplied in packed form,


   ``x`` is a vector of length ``n``.


hpr (Buffer Version)
--------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. cpp:function::  void onemkl::blas::hpr(sycl::queue &queue, uplo upper_lower, std::int64_t n, T alpha, sycl::buffer<T,1> &x, std::int64_t incx, sycl::buffer<T,1> &a)

.. container:: section


   .. rubric:: Input Parameters
      :class: sectiontitle


   queue
      The queue where the routine should be executed.


   upper_lower
      Specifies whether ``A`` is upper or lower triangular. See :ref:`onemkl_datatypes` for more details.


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
      (``n``\ \*(``n``-1))/2. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


      The imaginary part of the diagonal elements need not be set and
      are assumed to be zero.


.. container:: section


   .. rubric:: Output Parameters
      :class: sectiontitle


   a
      Buffer holding the updated upper triangular part of the Hermitian
      matrix ``A`` if ``upper_lower =upper``, or the updated lower
      triangular part of the Hermitian matrix ``A`` if
      ``upper_lower =lower``.


      The imaginary parts of the diagonal elements are set to zero.


hpr (USM Version)
-----------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. container:: dlsyntaxpara


         .. cpp:function::  sycl::event onemkl::blas::hpr(sycl::queue &queue, uplo upper_lower, std::int64_t n, T alpha, const T *x, std::int64_t incx, T *a, const sycl::vector_class<sycl::event> &dependencies = {})
   .. container:: section


      .. rubric:: Input Parameters
         :class: sectiontitle


      queue
         The queue where the routine should be executed.


      upper_lower
         Specifies whether ``A`` is upper or lower triangular. See :ref:`onemkl_datatypes` for more details.


      n
         Number of rows and columns of ``A``. Must be at least zero.


      alpha
         Scaling factor for the matrix-vector product.


      x
         Pointer to input vector ``x``. The array holding input vector
         ``x`` must be of size at least (1 + (``n`` - 1)*abs(``incx``)).
         See `Matrix and Vector
         Storage <../matrix-storage.html>`__ for
         more details.


      incx
         Stride of vector ``x``.


      a
         Pointer to input matrix ``A``. The array holding input matrix
         ``A`` must have size at least (``n``\ \*(``n``-1))/2. See
         `Matrix and Vector
         Storage <../matrix-storage.html>`__ for
         more details.


         The imaginary part of the diagonal elements need not be set and
         are assumed to be zero.


      dependencies
         List of events to wait for before starting computation, if any.
         If omitted, defaults to no dependencies.


   .. container:: section


      .. rubric:: Output Parameters
         :class: sectiontitle


      a
         Pointer to the updated upper triangular part of the Hermitian
         matrix ``A`` if ``upper_lower =upper``, or the updated lower
         triangular part of the Hermitian matrix ``A`` if
         ``upper_lower =lower``.


         The imaginary parts of the diagonal elements are set to zero.


   .. container:: section


      .. rubric:: Return Values
         :class: sectiontitle


      Output event to wait on to ensure computation is complete.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-2-routines`
