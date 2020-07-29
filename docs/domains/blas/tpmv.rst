.. _onemkl_blas_tpmv:

tpmv
====


.. container::


   Computes a matrix-vector product using a triangular packed matrix.



      ``tpmv`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``float`` 
         * -  ``double`` 
         * -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 




.. container:: section


   .. rubric:: Description
      :class: sectiontitle


   The ``tpmv`` routines compute a matrix-vector product with a triangular
   packed matrix. The operation is defined as


      x <- op(A)*x


   where:


   op(``A``) is one of op(``A``) = ``A``, or op(``A``) =
   ``A``\ :sup:`T`, or op(``A``) = ``A``\ :sup:`H`,


   ``A`` is an ``n``-by-``n`` unit or non-unit, upper or lower
   triangular band matrix, supplied in packed form,


   ``x`` is a vector of length ``n``.


tpmv (Buffer Version)
---------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. cpp:function::  void oneapi::mkl::blas::tpmv(sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_nonunit, std::int64_t n, sycl::buffer<T,1> &a, sycl::buffer<T,1> &x, std::int64_t incx)

.. container:: section


   .. rubric:: Input Parameters
      :class: sectiontitle


   queue
      The queue where the routine should be executed.


   upper_lower
      Specifies whether ``A`` is upper or lower triangular. See :ref:`onemkl_datatypes` for more details.


   trans
      Specifies op(``A``), the transposition operation applied to ``A``. See :ref:`onemkl_datatypes` for more details.


   unit_nonunit
      Specifies whether the matrix ``A`` is unit triangular or not. See :ref:`onemkl_datatypes` for more details.


   n
      Numbers of rows and columns of ``A``. Must be at least zero.


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


.. container:: section


   .. rubric:: Output Parameters
      :class: sectiontitle


   x
      Buffer holding the updated vector ``x``.


tpmv (USM Version)
------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. container:: dlsyntaxpara


         .. cpp:function::  sycl::event oneapi::mkl::blas::tpmv(sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_nonunit, std::int64_t n, const T *a, T *x, std::int64_t incx, const sycl::vector_class<sycl::event> &dependencies = {})
   .. container:: section


      .. rubric:: Input Parameters
         :class: sectiontitle


      queue
         The queue where the routine should be executed.


      upper_lower
         Specifies whether ``A`` is upper or lower triangular. See :ref:`onemkl_datatypes` for more details.


      trans
         Specifies op(``A``), the transposition operation applied to
         ``A``. See :ref:`onemkl_datatypes` for more details.


      unit_nonunit
         Specifies whether the matrix ``A`` is unit triangular or not. See :ref:`onemkl_datatypes` for more details.


      n
         Numbers of rows and columns of ``A``. Must be at least zero.


      a
         Pointer to input matrix ``A``. The array holding input matrix
         ``A`` must have size at least (``n``\ \*(``n``\ +1))/2. See
         `Matrix and Vector
         Storage <../matrix-storage.html>`__ for
         more details.


      x
         Pointer to input vector ``x``. The array holding input vector
         ``x`` must be of size at least (1 + (``n`` - 1)*abs(``incx``)).
         See `Matrix and Vector
         Storage <../matrix-storage.html>`__ for
         more details.


      incx
         Stride of vector ``x``.


      dependencies
         List of events to wait for before starting computation, if any.
         If omitted, defaults to no dependencies.


   .. container:: section


      .. rubric:: Output Parameters
         :class: sectiontitle


      x
         Pointer to the updated vector ``x``.


   .. container:: section


      .. rubric:: Return Values
         :class: sectiontitle


      Output event to wait on to ensure computation is complete.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-2-routines`
