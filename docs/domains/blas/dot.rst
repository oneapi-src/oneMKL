.. _onemkl_blas_dot:

dot
===


.. container::


   Computes the dot product of two real vectors.



      ``dot`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
           -  T_res 
         * -  ``float`` 
           -  ``float`` 
         * -  ``double`` 
           -  ``double`` 
         * -  ``float`` 
           -  ``double`` 




.. container:: section


   .. rubric:: Description
      :class: sectiontitle


   The dot routines perform a dot product between two vectors:


   |image0|


   .. container:: Note


      .. rubric:: Note
         :class: NoteTipHead


      For the mixed precision version (inputs are float while result is
      double), the dot product is computed with double precision.


dot (Buffer Version)
--------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. cpp:function::  void oneapi::mkl::blas::dot(sycl::queue &queue, std::int64_t n, sycl::buffer<T,1> &x, std::int64_t incx, sycl::buffer<T,1> &y, std::int64_t incy, sycl::buffer<T_res,1> &result)
.. container:: section


   .. rubric:: Input Parameters
      :class: sectiontitle


   queue
      The queue where the routine should be executed.


   n
      Number of elements in vectors ``x`` and ``y``.


   x
      Buffer holding input vector ``x``. The buffer must be of size at least
      ``(1 + (n – 1)*abs(incx))``. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   incx
      Stride of vector ``x``.


   y
      Buffer holding input vector ``y``. The buffer must be of size at least
      ``(1 + (n – 1)*abs(incy))``. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   incy
      Stride of vector ``y``.


.. container:: section


   .. rubric:: Output Parameters
      :class: sectiontitle


   result
      Buffer where the result (a scalar) will be stored.


dot (USM Version)
-----------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. container:: dlsyntaxpara


         .. cpp:function::  sycl::event oneapi::mkl::blas::dot(sycl::queue &queue, std::int64_t n, const T *x, std::int64_t incx, const T *y, std::int64_t incy, T_res *result, const sycl::vector_class<sycl::event> &dependencies = {})
   .. container:: section


      .. rubric:: Input Parameters
         :class: sectiontitle


      queue
         The queue where the routine should be executed.


      n
         Number of elements in vectors ``x`` and ``y``.


      x
         Pointer to the input vector ``x``. The array holding the vector ``x``
         must be of size at least ``(1 + (n – 1)*abs(incx))``. See
         `Matrix and Vector
         Storage <../matrix-storage.html>`__ for
         more details.


      incx
         Stride of vector ``x``.


      y
         Pointer to the input vector ``y``. The array holding the vector ``y``
         must be of size at least ``(1 + (n – 1)*abs(incy))``. See
         `Matrix and Vector
         Storage <../matrix-storage.html>`__ for
         more details.


      incy
         Stride of vector ``y``.


      dependencies
         List of events to wait for before starting computation, if any.
         If omitted, defaults to no dependencies.


   .. container:: section


      .. rubric:: Output Parameters
         :class: sectiontitle


      result
         Pointer to where the result (a scalar) will be stored.


   .. container:: section


      .. rubric:: Return Values
         :class: sectiontitle


      Output event to wait on to ensure computation is complete.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-1-routines`
.. |image0| image:: ../equations/GUID-75532DED-BE44-4D85-B9C0-99C825778ee1.png
   :class: img-middle

