.. _rot:

rot
===


.. container::


   Performs rotation of points in the plane.


   .. container:: section
      :name: GUID-9DD44991-6A55-49EE-BD0C-F13406FFBE52


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void rot(queue &exec_queue, std::int64_t n,      buffer<T,1> &x, std::int64_t incx, buffer<T,1> &y, std::int64_t      incy, T_scalar c, T_scalar s)

      ``rot`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
           -  T_scalar 
         * -  ``float`` 
           -  ``float`` 
         * -  ``double`` 
           -  ``double`` 
         * -  ``std::complex<float>`` 
           -  ``float`` 
         * -  ``std::complex<double>`` 
           -  ``double`` 




.. container:: section
   :name: GUID-8B7F46D1-5047-4D4C-AF66-F0A3E4AC2BA5


   .. rubric:: Description
      :name: description
      :class: sectiontitle


   Given two vectors ``x`` and ``y`` of ``n`` elements, the rot routines
   compute four scalar-vector products and update the input vectors with
   the sum of two of these scalar-vector products as follow:

  
   x <- c*x + s*y

   y <- c*y - s*x
  


.. container:: section
   :name: GUID-A615800D-734E-4997-BB91-1C76AEEE9EC2


   .. rubric:: Input Parameters
      :name: input-parameters
      :class: sectiontitle


   exec_queue
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


   y
      Buffer holding input vector ``y``. The buffer must be of size at
      least (1 + (``n`` - 1)*abs(``incy``)). See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   incy
      Stride of vector y.


   c
      Scaling factor.


   s
      Scaling factor.


.. container:: section
   :name: GUID-2B160DEB-ADBB-4044-8078-4B613A0DA4E1


   .. rubric:: Output Parameters
      :name: output-parameters
      :class: sectiontitle


   x
      Buffer holding updated buffer ``x``.


   y
      Buffer holding updated buffer ``y``.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-1-routines`
      


.. container::

