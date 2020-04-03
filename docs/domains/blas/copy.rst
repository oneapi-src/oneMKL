.. _copy:

copy
====


.. container::


   Copies a vector to another vector.


   .. container:: section
      :name: GUID-D6B6C72E-9516-40C9-B034-9F344C41AAF3


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void copy(queue &exec_queue, std::int64_t n,      buffer<T,1> &x, std::int64_t incx, buffer<T,1> &y, std::int64_t      incy)

      ``copy`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``float`` 
         * -  ``double`` 
         * -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 




.. container:: section
   :name: GUID-5E0A9C5F-BDD5-41E6-97CD-4316FD58C347


   .. rubric:: Description
      :name: description
      :class: sectiontitle


   The copy routines copy one vector to another:


  


      y ←x


   where x and y are vectors of n elements.


.. container:: section
   :name: GUID-6F86EF6A-8FFE-4C6A-8B71-23B95C1F1365


   .. rubric:: Input Parameters
      :name: input-parameters
      :class: sectiontitle


   exec_queue
      The queue where the routine should be executed.


   n
      Number of elements in vector x.


   x
      Buffer holding input vector x. The buffer must be of size at least
      ``(1 + (n – 1)*abs(incx))``. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   incx
      Stride of vector x.


   incy
      Stride of vector y.


.. container:: section
   :name: GUID-4ABB603B-835C-428B-B880-2F088BAB5456


   .. rubric:: Output Parameters
      :name: output-parameters
      :class: sectiontitle


   y
      Buffer holding the updated vector y.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-1-routines`
      


.. container::

