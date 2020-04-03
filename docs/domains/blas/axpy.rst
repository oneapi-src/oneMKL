.. _axpy:

axpy
====


.. container::


   Computes a vector-scalar product and adds the result to a vector.


   .. container:: section
      :name: GUID-17ADB23B-C9B0-44B4-89F9-B7199DA9E872


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void axpy(queue &exec_queue, std::int64_t n, T      alpha, buffer<T,1> &x, std::int64_t incx, buffer<T,1> &y,      std::int64_t incy)

      ``axpy`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``float`` 
         * -  ``double`` 
         * -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 




.. container:: section
   :name: GUID-4BC6BF9A-BAB9-4078-A6B5-9C7ECB9D4821


   .. rubric:: Description
      :name: description
      :class: sectiontitle


   The axpy routines compute a scalar-vector product and add the result
   to a vector:


  


      y <- alpha*x+y


   where:


   ``x`` and ``y`` are vectors of ``n`` elements,


   ``alpha`` is a scalar.


.. container:: section
   :name: GUID-6F86EF6A-8FFE-4C6A-8B71-23B95C1F1365


   .. rubric:: Input Parameters
      :name: input-parameters
      :class: sectiontitle


   exec_queue
      The queue where the routine should be executed.


   n
      Number of elements in vector x.


   alpha
      Specifies the scalar alpha.


   x
      Buffer holding input vector x. The buffer must be of size at least
      ``(1 + (n – 1)*abs(incx))``. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   incx
      Stride of vector x.


   y
      Buffer holding input vector y. The buffer must be of size at least
      ``(1 + (n – 1)*abs(incy))``. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   incy
      Stride of vector y.


.. container:: section
   :name: GUID-A0926D96-B673-48A4-986A-033719589288


   .. rubric:: Output Parameters
      :name: output-parameters
      :class: sectiontitle


   y
      Buffer holding the updated vector y.



.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-1-routines`
      


.. container::

