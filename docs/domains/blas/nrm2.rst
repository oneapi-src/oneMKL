.. _nrm2:

nrm2
====


.. container::


   Computes the Euclidean norm of a vector.


   .. container:: section
      :name: GUID-F55A15D5-CCDA-4C44-B86F-C9A5FB36725E


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void nrm2(queue &exec_queue, std::int64_t n,      buffer<T,1> &x, std::int64_t incx, buffer<T_res,1> &result)

      ``nrm2`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
           -  T_res 
         * -  ``float`` 
           -  ``float`` 
         * -  ``double`` 
           -  ``double`` 
         * -  ``std::complex<float>`` 
           -  ``float`` 
         * -  ``std::complex<double>`` 
           -  ``double`` 




.. container:: section
   :name: GUID-2BF2C965-5A8C-47F1-9C73-FB0E485CE32A


   .. rubric:: Description
      :name: description
      :class: sectiontitle


   The nrm2 routines computes Euclidean norm of a vector


  


      result = ||x||,


   where:


   ``x`` is a vector of ``n`` elements.


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
      Buffer where the Euclidean norm of the vector ``x`` will be
      stored.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-1-routines`
      


.. container::

