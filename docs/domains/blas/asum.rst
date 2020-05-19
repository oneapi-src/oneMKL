.. _onemkl_blas_asum:

asum
====


.. container::


   Computes the sum of magnitudes of the vector elements.



      ``asum`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
           -  T_res 
         * -  ``float`` 
           -  ``float`` 
         * -  ``double`` 
           -  ``double`` 
         * -  ``std::complex<float>`` 
           -  ``float`` 
         * -  ``std::complex<double>`` 
           -  ``double`` 




.. container:: section


   .. rubric:: Description
      :class: sectiontitle


   The asum routine computes the sum of the magnitudes of elements of a
   real vector, or the sum of magnitudes of the real and imaginary parts
   of elements of a complex vector:


   |image0|


   where ``x`` is a vector with ``n`` elements.


asum (Buffer Version)
---------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. cpp:function::  void onemkl::blas::asum(sycl::queue &queue, std::int64_t n, sycl::buffer<T,1> &x, std::int64_t incx, sycl::buffer<T_res,1> &result)
.. container:: section


   .. rubric:: Input Parameters
      :class: sectiontitle


   queue
      The queue where the routine should be executed.


   n
      Number of elements in vector ``x``.


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


   result
      Buffer where the scalar result is stored (the sum of magnitudes of
      the real and imaginary parts of all elements of the vector).


asum (USM Version)
------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. container:: dlsyntaxpara


         .. cpp:function::  sycl::event onemkl::blas::asum(sycl::queue &queue, std::int64_t n, const T *x, std::int64_t incx, T_res *result, const sycl::vector_class<sycl::event> &dependencies = {})
   .. container:: section


      .. rubric:: Input Parameters
         :class: sectiontitle


      queue
         The queue where the routine should be executed.


      n
         Number of elements in vector ``x``.


      x
         Pointer to input vector ``x``. The array holding the vector
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


      result
         Pointer to the output matrix where the scalar result is stored
         (the sum of magnitudes of the real and imaginary parts of all
         elements of the vector).


   .. container:: section


      .. rubric:: Return Values
         :class: sectiontitle


      Output event to wait on to ensure computation is complete.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-1-routines`
.. |image0| image:: ../equations/GUID-4F76F5A1-251F-4AC0-A2E0-A3B4B6F39ee1.png
   :class: img-middle

