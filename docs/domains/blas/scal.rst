.. _onemkl_blas_scal:

scal
====


.. container::


   Computes the product of a vector by a scalar.



      ``scal`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
           -  T_scalar 
         * -  ``float`` 
           -  ``float`` 
         * -  ``double`` 
           -  ``double`` 
         * -  ``std::complex<float>`` 
           -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 
           -  ``std::complex<double>`` 
         * -  ``std::complex<float>`` 
           -  ``float`` 
         * -  ``std::complex<double>`` 
           -  ``double`` 




.. container:: section


   .. rubric:: Description
      :class: sectiontitle


   The ``scal`` routines computes a scalar-vector product:


      x <- alpha*x


   where:


   ``x`` is a vector of ``n`` elements,


   ``alpha`` is a scalar.


scal (Buffer Version)
---------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. cpp:function::  void oneapi::mkl::blas::scal(sycl::queue &queue, std::int64_t n, T_scalar alpha, sycl::buffer<T,1> &x, std::int64_t incx)

.. container:: section


   .. rubric:: Input Parameters
      :class: sectiontitle


   queue
      The queue where the routine should be executed.


   n
      Number of elements in vector ``x``.


   alpha
      Specifies the scalar ``alpha``.


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
      Buffer holding updated buffer ``x``.


scal (USM Version)
------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. container:: dlsyntaxpara


         .. cpp:function::  sycl::event oneapi::mkl::blas::scal(sycl::queue &queue, std::int64_t n, T_scalar alpha, T *x, std::int64_t incx, const         sycl::vector_class<sycl::event> &dependencies = {})
   .. container:: section


      .. rubric:: Input Parameters
         :class: sectiontitle


      queue
         The queue where the routine should be executed.


      n
         Number of elements in vector ``x``.


      alpha
         Specifies the scalar ``alpha``.


      x
         Pointer to the input vector ``x``. The array must be of size at
         least (1 + (``n`` - 1)*abs(``incx``)). See `Matrix and Vector
         Storage <../matrix-storage.html>`__ for
         more details.


      incx
         Stride of vector ``x``.


   .. container:: section


      .. rubric:: Output Parameters
         :class: sectiontitle


      x
         Pointer to the updated array ``x``.


   .. container:: section


      .. rubric:: Return Values
         :class: sectiontitle


      Output event to wait on to ensure computation is complete.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-1-routines`
