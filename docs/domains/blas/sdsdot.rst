.. _sdsdot:

sdsdot
======


.. container::


   Computes a vector-vector dot product with double precision.


   .. container:: section
      :name: GUID-2DDFDC38-65FA-40F5-AACB-8E383623EF4A


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void sdsdot(queue &exec_queue, std::int64_t n,      float sb, buffer<float,1> &x, std::int64_t incx, buffer<float,1>      &y, std::int64_t incy, buffer<float,1> &result)

      .. rubric:: Description
         :name: description
         :class: sectiontitle


      The sdsdot routines perform a dot product between two vectors with
      double precision:


      |image0|


      .. rubric:: Input Parameters
         :name: input-parameters
         :class: sectiontitle


      exec_queue
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
         Stride of vector x.


      y
         Buffer holding input vector ``y``. The buffer must be of size
         at least (1 + (``n`` - 1)*abs(``incxy``)). See `Matrix and
         Vector
         Storage <../matrix-storage.html>`__ for
         more details.


      incy
         Stride of vector y.


      .. rubric:: Output Parameters
         :name: output-parameters
         :class: sectiontitle


      result
         Buffer where the result (a scalar) will be stored. If ``n`` < 0
         the result is ``sb``.


      **Parent topic:** :ref:`blas-level-1-routines`
      


.. |image0| image:: ../equations/GUID-9DB212E1-03E2-430C-8B1F-8F5CBD4F2ee1.png
   :class: img-middle

