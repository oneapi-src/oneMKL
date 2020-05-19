.. _onemkl_blas_dotc:

dotc
====


.. container::


   Computes the dot product of two complex vectors, conjugating the
   first vector.



      ``dotc`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 




.. container:: section


   .. rubric:: Description
      :class: sectiontitle


   The ``dotc`` routines perform a dot product between two complex
   vectors, conjugating the first of them:


   |image0|


dotc (Buffer Version)
---------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. cpp:function::  void onemkl::blas::dotc(sycl::queue &queue, std::int64_t n, sycl::buffer<T,1> &x, std::int64_t incx, sycl::buffer<T,1> &y, std::int64_t incy, sycl::buffer<T,1> &result)
.. container:: section


   .. rubric:: Input Parameters
      :class: sectiontitle


   queue
      The queue where the routine should be executed.


   n
      The number of elements in vectors ``x`` and ``y``.


   x
      Buffer holding input vector ``x``. The buffer must be of size at
      least (1 + (``n`` - 1)*abs(``incx``)). See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   incx
      The stride of vector ``x``.


   y
      Buffer holding input vector ``y``. The buffer must be of size at
      least (1 + (``n`` - 1)*abs(``incy``)). See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details..


   incy
      The stride of vector ``y``.


.. container:: section


   .. rubric:: Output Parameters
      :class: sectiontitle


   result
      The buffer where the result (a scalar) is stored.


dotc (USM Version)
------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. container:: dlsyntaxpara


         .. cpp:function::  void onemkl::blas::dotc(sycl::queue &queue, std::int64_t n, const T *x, std::int64_t incx, const T *y, std::int64_t incy, T *result, const sycl::vector_class<sycl::event> &dependencies = {})
   .. container:: section


      .. rubric:: Input Parameters
         :class: sectiontitle


      queue
         The queue where the routine should be executed.


      n
         The number of elements in vectors ``x`` and ``y``.


      x
         Pointer to input vector ``x``. The array holding the input
         vector ``x`` must be of size at least (1 + (``n`` -
         1)*abs(``incx``)). See `Matrix and Vector
         Storage <../matrix-storage.html>`__ for
         more details.


      incx
         The stride of vector ``x``.


      y
         Pointer to input vector ``y``. The array holding the input
         vector ``y`` must be of size at least (1 + (``n`` -
         1)*abs(``incy``)). See `Matrix and Vector
         Storage <../matrix-storage.html>`__ for
         more details..


      incy
         The stride of vector ``y``.


      dependencies
         List of events to wait for before starting computation, if any.
         If omitted, defaults to no dependencies.


   .. container:: section


      .. rubric:: Output Parameters
         :class: sectiontitle


      result
         The pointer to where the result (a scalar) is stored.


   .. container:: section


      .. rubric:: Return Values
         :class: sectiontitle


      Output event to wait on to ensure computation is complete.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-1-routines`
.. |image0| image:: ../equations/GUID-B2211D34-A472-4FB8-9CFB-1E11AF4F0ee1.png
   :class: img-middle

