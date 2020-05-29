.. _onemkl_blas_sdsdot:

sdsdot
======


.. container::


   Computes a vector-vector dot product with double precision.



   .. container:: section


      .. rubric:: Description
         :class: sectiontitle


      The ``sdsdot`` routines perform a dot product between two vectors with
      double precision:


      |image0|

sdsdot (Buffer Version)
-----------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. cpp:function::  void onemkl::blas::sdsdot(sycl::queue &queue, std::int64_t n, float sb, sycl::buffer<float,1> &x, std::int64_t incx, sycl::buffer<float,1> &y, std::int64_t incy, sycl::buffer<float,1> &result)
   .. container:: section
   
   
      .. rubric:: Input Parameters
         :class: sectiontitle


      queue
         The queue where the routine should be executed.


      n
         Number of elements in vectors ``x`` and ``y``.


      sb
         Single precision scalar to be added to the dot product.


      x
         Buffer holding input vector ``x``. The buffer must be of size
         at least (1 + (``n`` - 1)*abs(``incx``)). See `Matrix and
         Vector
         Storage <../matrix-storage.html>`__ for
         more details.


      incx
         Stride of vector ``x``.


      y
         Buffer holding input vector ``y``. The buffer must be of size
         at least (1 + (``n`` - 1)*abs(``incxy``)). See `Matrix and
         Vector
         Storage <../matrix-storage.html>`__ for
         more details.


      incy
         Stride of vector ``y``.

   .. container:: section
   
   
      .. rubric:: Output Parameters
         :class: sectiontitle


      result
         Buffer where the result (a scalar) will be stored. If ``n`` < 0
         the result is ``sb``.

sdsdot (USM Version)
--------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. container:: dlsyntaxpara


         .. cpp:function::  sycl::event onemkl::blas::sdsdot(sycl::queue &queue, std::int64_t n, float sb, const float *x, std::int64_t incx, const float *y, std::int64_t incy, float *result, const sycl::vector_class<sycl::event> &dependencies = {})
      .. container:: section
      
      
         .. rubric:: Input Parameters
            :class: sectiontitle


         queue
            The queue where the routine should be executed.


         n
            Number of elements in vectors ``x`` and ``y``.


         sb
            Single precision scalar to be added to the dot product.


         x
            Pointer to the input vector ``x``. The array must be of size
            at least (1 + (``n`` - 1)*abs(``incx``)). See `Matrix and
            Vector
            Storage <../matrix-storage.html>`__
            for more details.


         incx
            Stride of vector ``x``.


         y
            Pointer to the input vector ``y``. The array must be of size
            at least (1 + (``n`` - 1)*abs(``incxy``)). See `Matrix and
            Vector
            Storage <../matrix-storage.html>`__
            for more details.


         incy
            Stride of vector ``y``.


         dependencies
            List of events to wait for before starting computation, if
            any. If omitted, defaults to no dependencies.

      .. container:: section
      
      
         .. rubric:: Output Parameters
            :class: sectiontitle


         result
            Pointer to where the result (a scalar) will be stored. If
            ``n`` < 0 the result is ``sb``.

      .. container:: section
   
   
         .. rubric:: Return Values
            :class: sectiontitle


         Output event to wait on to ensure computation is complete.

.. container:: familylinks


   .. container:: parentlink

         **Parent topic:** :ref:`blas-level-1-routines`
.. |image0| image:: ../equations/GUID-9B91DAAE-72DD-4799-9983-12B021993ee1.png
   :class: img-middle

