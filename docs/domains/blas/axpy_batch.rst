.. _onemkl_blas_axpy_batch:

axpy_batch
==========

.. container::


   The ``axpy_batch`` routines are batched versions of `axpy <axpy.html>`__, performing
   multiple ``axpy`` operations in a single call. Each ``axpy`` 
   operation adds a scalar-vector product to a vector.
   

      ``axpy_batch`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``float`` 
         * -  ``double`` 
         * -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 



axpy_batch (USM Version)
------------------------

.. container:: section


   .. rubric:: Description
      :class: sectiontitle


   The USM version of ``axpy_batch`` supports group API. 

   The group API operation is defined as
  
   ::
      
      idx = 0
      for i = 0 … group_count – 1
          for j = 0 … group_size – 1
              X and Y are vectors in x[idx] and y[idx]
              Y := alpha[i] * X + Y
              idx := idx + 1
          end for
      end for


   where:

   ``alpha`` is scalar

   ``X`` and ``Y`` are vectors.


   For group API, ``x`` and ``y`` arrays contain the pointers for all the input vectors. 
   The total number of vectors in ``x`` and ``y`` are given by:

      total_batch_count = sum of all of the group_size entries


   **Group API**

.. container:: section


   .. rubric:: Syntax
      :class: sectiontitle


   .. container:: dlsyntaxpara


      .. cpp:function::  sycl::event oneapi::mkl::blas::axpy_batch(sycl::queue &queue, std::int64_t *n, T *alpha, const T **x, std::int64_t *incx, T **y, std::int64_t *incy, std::int64_t group_count, std::int64_t *group_size, const sycl::vector_class<sycl::event> &dependencies = {})

    
.. container:: section


   .. rubric:: Input Parameters
      :class: sectiontitle

   queue
      The queue where the routine should be executed.

   n
      Array of ``group_count`` integers. ``n[i]`` specifies the number of elements in vectors ``X`` and ``Y`` for every vector in group ``i``.


   alpha
       Array of ``group_count`` scalar elements. ``alpha[i]`` specifies the scaling factor for vector ``X`` in group ``i``.


   x
      Array of pointers to input vectors ``X`` with size ``total_batch_count``.
      The size of array allocated for the ``X`` vector of the group ``i`` must be at least ``(1 + (n[i] – 1)*abs(incx[i]))``. 
      See `Matrix and Vector Storage <../matrix-storage.html>`__ for more details.

   incx
      Array of ``group_count`` integers. ``incx[i]`` specifies the stride of vector ``X`` in group ``i``.
 
   y
      Array of pointers to input/output vectors ``Y`` with size ``total_batch_count``.
      The size of array allocated for the ``Y`` vector of the group ``i`` must be at least ``(1 + (n[i] – 1)*abs(incy[i]))``. 
      See `Matrix and Vector Storage <../matrix-storage.html>`__ for more details.

   incy
      Array of ``group_count`` integers. ``incy[i]`` specifies the stride of vector ``Y`` in group ``i``.


   group_count
      Number of groups. Must be at least 0.


   group_size
      Array of ``group_count`` integers. ``group_size[i]`` specifies the number of ``axpy`` operations in group ``i``. 
      Each element in ``group_size`` must be at least 0.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.


.. container:: section


   .. rubric:: Output Parameters
      :class: sectiontitle


   y
      Array of pointers holding the ``Y`` vectors, overwritten by ``total_batch_count`` ``axpy`` operations of the form 
      ``alpha*X + Y``.


.. container:: section


   .. rubric:: Return Values
      :class: sectiontitle


   Output event to wait on to ensure computation is complete.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:**:ref:`blas-like-extensions`
      


