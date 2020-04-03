.. _scal:

scal
====


.. container::


   Computes the product of a vector by a scalar.


   .. container:: section
      :name: GUID-178A4C6A-3BA5-40F7-A3D6-4B6590B75EB4


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void scal(queue &exec_queue, std::int64_t n,      T_scalar alpha, buffer<T,1> &x, std::int64_t incx)

      ``scal`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
           -  T_scalar 
         * -  ``float`` 
           -  ``float`` 
         * -  ``double`` 
           -  ``double`` 
         * -  ``std::complex<float>`` 
           -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 
           -  ``std::complex<double>`` 
         * -  ``std::complex<float>`` 
           -  ``float`` 
         * -  ``std::complex<double>`` 
           -  ``double`` 




.. container:: section
   :name: GUID-8DDCA613-2750-43D0-A89B-13866F2DDE8C


   .. rubric:: Description
      :name: description
      :class: sectiontitle


   The scal routines computes a scalar-vector product:


  


      x <- alpha*x


   where:


   ``x`` is a vector of ``n`` elements,


   ``alpha`` is a scalar.


.. container:: section
   :name: GUID-A615800D-734E-4997-BB91-1C76AEEE9EC2


   .. rubric:: Input Parameters
      :name: input-parameters
      :class: sectiontitle


   exec_queue
      The queue where the routine should be executed.


   n
      Number of elements in vector ``x``.


   alpha
      Specifies the scalar ``alpha``.


   x
      Buffer holding input vector ``x``. The buffer must be of size at
      least (1 + (``n`` - 1)*abs(``incx``)). See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   incx
      Stride of vector x.


.. container:: section
   :name: GUID-B36EBB3E-C79B-49F8-9F47-7B19BD6BE105


   .. rubric:: Output Parameters
      :name: output-parameters
      :class: sectiontitle


   x
      Buffer holding updated buffer ``x``.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-1-routines`
      


.. container::

