.. _dot:

dot
===


.. container::


   Computes the dot product of two real vectors.


   .. container:: section
      :name: GUID-13355B56-0278-45E5-B310-3B0AC541C675


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void dot(queue &exec_queue, std::int64_t n,      buffer<T,1> &x, std::int64_t incx, buffer<T,1> &y, std::int64_t      incy, buffer<T_res,1> &result)

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
   :name: GUID-4BC6BF9A-BAB9-4078-A6B5-9C7ECB9D4821


   .. rubric:: Description
      :name: description
      :class: sectiontitle


   The dot routines perform a dot product between two vectors:


   |image0|


   .. container:: Note


      .. rubric:: Note
         :name: note
         :class: NoteTipHead


      For the mixed precision version (inputs are float while result is
      double), the dot product is computed with double precision.


.. container:: section
   :name: GUID-6F86EF6A-8FFE-4C6A-8B71-23B95C1F1365


   .. rubric:: Input Parameters
      :name: input-parameters
      :class: sectiontitle


   exec_queue
      The queue where the routine should be executed.


   n
      Number of elements in vectors x and y.


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
   :name: GUID-CAAFE234-AF82-4B61-8406-D57EC527BED5


   .. rubric:: Output Parameters
      :name: output-parameters
      :class: sectiontitle


   result
      Buffer where the result (a scalar) will be stored.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-1-routines`
      


.. container::


.. |image0| image:: ../equations/GUID-93DA36DC-40CA-4C01-B883-DABAB0D37ee1.png
   :class: img-middle

