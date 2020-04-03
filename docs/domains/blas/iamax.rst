.. _iamax:

iamax
=====


.. container::


   Finds the index of the element with the largest absolute value in a
   vector.


   .. container:: section
      :name: GUID-D1ABF76D-DB39-4C23-A217-EA2C7C6D1325


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void iamax(queue &exec_queue, std::int64_t n,      buffer<T, 1> &x, std::int64_t incx, buffer<std::int64_t, 1>      &result)

      iamax supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``float`` 
         * -  ``double`` 
         * -  ``std::complex<float>`` 
         * -  ``std:complex<double>`` 




.. container:: section
   :name: GUID-822D7950-256E-406D-9305-61F761080E69


   .. rubric:: Description
      :name: description
      :class: sectiontitle


   The iamax routines return an index ``i``\ such that ``x``\ [``i``]
   has the maximum absolute value of all elements in vector ``x`` (real
   variants), or such that ``|Re(x[i])| + |Im(x[i])|`` is maximal
   (complex variants).


   .. container:: Note


      .. rubric:: Note
         :name: note
         :class: NoteTipHead


      The index is zero-based.


   If either ``n`` or ``incx`` are not positive, the routine returns
   ``0``.


   If more than one vector element is found with the same largest
   absolute value, the index of the first one encountered is returned.


   If the vector contains ``NaN`` values, then the routine returns the
   index of the first ``NaN``.


.. container:: section
   :name: GUID-CE43FE84-2066-4095-BB7E-0691CD045443


   .. rubric:: Input Parameters
      :name: input-parameters
      :class: sectiontitle


   exec_queue
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
   :name: ARGUMENTS_EC9F05BE9B09443F8BC59207D5EA40F1


   .. rubric:: Output Parameters
      :name: output-parameters
      :class: sectiontitle


   result
      The buffer where the zero-based index ``i`` of the maximal element
      is stored.



.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-1-routines`
      


.. container::

