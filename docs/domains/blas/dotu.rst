.. _dotu:

dotu
====


.. container::


   Computes the dot product of two complex vectors.


   .. container:: section
      :name: GUID-27A695AE-7ED5-4CFF-9783-0E50D111BED2


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void dotu(queue &exec_queue, std::int64_t n,      buffer<T,1> &x, std::int64_t incx, buffer<T,1> &y, std::int64_t      incy, buffer<T,1> &result)

      ``dotu`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 




.. container:: section
   :name: GUID-7E67CFC6-917F-41A3-A664-F99EE4E04E43


   .. rubric:: Description
      :name: description
      :class: sectiontitle


   The dotu routines perform a dot product between two complex vectors:


   |image0|


.. container:: section
   :name: GUID-A615800D-734E-4997-BB91-1C76AEEE9EC2


   .. rubric:: Input Parameters
      :name: input-parameters
      :class: sectiontitle


   exec_queue
      The queue where the routine should be executed.


   n
      Number of elements in vectors ``x`` and ``y``.


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


.. container:: section
   :name: GUID-2B160DEB-ADBB-4044-8078-4B613A0DA4E1


   .. rubric:: Output Parameters
      :name: output-parameters
      :class: sectiontitle


   result
      Buffer where the result (a scalar) is stored.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-1-routines`
      


.. container::


.. |image0| image:: ../equations/GUID-3605ACD9-02D1-46D7-B791-F2F76F0D9ee1.png
   :class: img-middle

