.. _onemkl_blas_iamax:

iamax
=====


.. container::


   Finds the index of the element with the largest absolute value in a
   vector.



      ``iamax`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``float`` 
         * -  ``double`` 
         * -  ``std::complex<float>`` 
         * -  ``std:complex<double>`` 




.. container:: section


   .. rubric:: Description
      :class: sectiontitle


   The ``iamax`` routines return an index ``i``\ such that ``x``\ [``i``]
   has the maximum absolute value of all elements in vector ``x`` (real
   variants), or such that ``|Re(x[i])| + |Im(x[i])|`` is maximal
   (complex variants).


   .. container:: Note


      .. rubric:: Note
         :class: NoteTipHead


      The index is zero-based.


   If either ``n`` or ``incx`` are not positive, the routine returns
   ``0``.


   If more than one vector element is found with the same largest
   absolute value, the index of the first one encountered is returned.


   If the vector contains ``NaN`` values, then the routine returns the
   index of the first ``NaN``.


iamax (Buffer Version)
----------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. cpp:function::  void oneapi::mkl::blas::iamax(sycl::queue &queue, std::int64_t n, sycl::buffer<T, 1> &x, std::int64_t incx, sycl::buffer<std::int64_t, 1> &result)

.. container:: section


   .. rubric:: Input Parameters
      :class: sectiontitle


   queue
      The queue where the routine should be executed.


   n
      The number of elements in vector ``x``.


   x
      The buffer that holds the input vector ``x``. The buffer must be
      of size at least (1 + (``n`` - 1)*abs(``incx``)). See `Matrix and
      Vector Storage <../matrix-storage.html>`__
      for more details.


   incx
      The stride of vector ``x``.


.. container:: section


   .. rubric:: Output Parameters
      :class: sectiontitle


   result
      The buffer where the zero-based index ``i`` of the maximal element
      is stored.


iamax (USM Version)
-------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. container:: dlsyntaxpara


         .. cpp:function::  sycl::event oneapi::mkl::blas::iamax(sycl::queue &queue, std::int64_t n, const T *x, std::int64_t incx, T_res *result, const sycl::vector_class<sycl::event> &dependencies = {})
   .. container:: section


      .. rubric:: Input Parameters
         :class: sectiontitle


      queue
         The queue where the routine should be executed.


      n
         The number of elements in vector ``x``.


      x
         The pointer to the input vector ``x``. The array holding the
         input vector ``x`` must be of size at least (1 + (``n`` -
         1)*abs(``incx``)). See `Matrix and Vector
         Storage <../matrix-storage.html>`__ for
         more details.


      incx
         The stride of vector ``x``.


      dependencies
         List of events to wait for before starting computation, if any.
         If omitted, defaults to no dependencies.


   .. container:: section


      .. rubric:: Output Parameters
         :class: sectiontitle


      result
         The pointer to where the zero-based index ``i`` of the maximal
         element is stored.


   .. container:: section


      .. rubric:: Return Values
         :class: sectiontitle


      Output event to wait on to ensure computation is complete.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-1-routines`
