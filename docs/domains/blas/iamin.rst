.. _onemkl_blas_iamin:

iamin
=====


.. container::


   Finds the index of the element with the smallest absolute value.



      ``iamin`` supports the following precisions.


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


   The ``iamin`` routines return an index ``i`` such that ``x``\ [``i``] has
   the minimum absolute value of all elements in vector ``x`` (real
   variants), or such that \|Re(``x``\ [``i``])\| +
   \|Im(``x``\ [``i``])\| is maximal (complex variants).


   .. container:: Note


      .. rubric:: Note
         :class: NoteTipHead


      The index is zero-based.


   If either ``n`` or ``incx`` are not positive, the routine returns
   ``0``.


   If more than one vector element is found with the same smallest
   absolute value, the index of the first one encountered is returned.


   If the vector contains ``NaN`` values, then the routine returns the
   index of the first ``NaN``.


iamin (Buffer Version)
----------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. cpp:function::  void onemkl::blas::iamin(sycl::queue &queue, std::int64_t n,      sycl::buffer<T,1> &x, std::int64_t incx, sycl::buffer<std::int64_t,1>      &result)
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
      Stride of vector x.


.. container:: section


   .. rubric:: Output Parameters
      :class: sectiontitle


   result
      Buffer where the zero-based index ``i`` of the minimum element
      will be stored.


iamin (USM Version)
-------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. container:: dlsyntaxpara


         .. cpp:function::  sycl::event onemkl::blas::iamin(sycl::queue &queue, std::int64_t n, const T *x, std::int64_t incx, T_res *result, const sycl::vector_class<sycl::event> &dependencies = {})
   .. container:: section


      .. rubric:: Input Parameters
         :class: sectiontitle


      queue
         The queue where the routine should be executed.


      n
         Number of elements in vector ``x``.


      x
         The pointer to input vector ``x``. The array holding input
         vector ``x`` must be of size at least (1 + (``n`` -
         1)*abs(``incx``)). See `Matrix and Vector
         Storage <../matrix-storage.html>`__ for
         more details.


      incx
         Stride of vector x.


   .. container:: section


      .. rubric:: Output Parameters
         :class: sectiontitle


      result
         Pointer to where the zero-based index ``i`` of the minimum
         element will be stored.


   .. container:: section


      .. rubric:: Return Values
         :class: sectiontitle


      Output event to wait on to ensure computation is complete.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-1-routines`
