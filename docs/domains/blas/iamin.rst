.. _iamin:

iamin
=====


.. container::


   Finds the index of the element with the smallest absolute value.


   .. container:: section
      :name: GUID-5D077B60-17B5-4961-AFF7-20D78BFB2A07


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void iamin(queue &exec_queue, std::int64_t n,      buffer<T,1> &x, std::int64_t incx, buffer<std::int64_t,1>      &result)

      ``iamin`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``float`` 
         * -  ``double`` 
         * -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 




.. container:: section
   :name: GUID-A820CE7B-E983-4D8F-A73A-753FD95BD507


   .. rubric:: Description
      :name: description
      :class: sectiontitle


   The iamin routines return an index ``i`` such that ``x``\ [``i``] has
   the minimum absolute value of all elements in vector ``x`` (real
   variants), or such that \|Re(``x``\ [``i``])\| +
   \|Im(``x``\ [``i``])\| is maximal (complex variants).


   .. container:: Note


      .. rubric:: Note
         :name: note
         :class: NoteTipHead


      The index is zero-based.


   If either ``n`` or ``incx`` are not positive, the routine returns
   ``0``.


   If more than one vector element is found with the same smallest
   absolute value, the index of the first one encountered is returned.


   If the vector contains ``NaN`` values, then the routine returns the
   index of the first ``NaN``.


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


.. container:: section
   :name: GUID-2B160DEB-ADBB-4044-8078-4B613A0DA4E1


   .. rubric:: Output Parameters
      :name: output-parameters
      :class: sectiontitle


   result
      Buffer where the zero-based index ``i`` of the minimum element
      will be stored.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-1-routines`
      


.. container::

