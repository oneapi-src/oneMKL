.. _onemkl_blas_rotg:

rotg
====


.. container::


   Computes the parameters for a Givens rotation.



      ``rotg`` supports the following precisions.


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


   .. rubric:: Description
      :class: sectiontitle


   Given the Cartesian coordinates ``(a, b)`` of a point, the ``rotg``
   routines return the parameters ``c``, ``s``, ``r``, and ``z``
   associated with the Givens rotation. The parameters ``c`` and ``s``
   define a unitary matrix such that:


   The parameter ``z`` is defined such that if \|\ ``a``\ \| >
   \|\ ``b``\ \|, ``z`` is ``s``; otherwise if ``c`` is not 0 ``z`` is
   1/``c``; otherwise ``z`` is 1.


rotg (Buffer Version)
---------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. cpp:function::  void onemkl::blas::rotg(sycl::queue &queue, sycl::buffer<T,1> &a,      sycl::buffer<T,1> &b, sycl::buffer<T_real,1> &c, sycl::buffer<T,1> &s)
.. container:: section


   .. rubric:: Input Parameters
      :class: sectiontitle


   queue
      The queue where the routine should be executed


   a
      Buffer holding the ``x``-coordinate of the point.


   b
      Buffer holding the ``y``-coordinate of the point.


.. container:: section


   .. rubric:: Output Parameters
      :class: sectiontitle


   a
      Buffer holding the parameter ``r`` associated with the Givens
      rotation.


   b
      Buffer holding the parameter ``z`` associated with the Givens
      rotation.


   c
      Buffer holding the parameter ``c`` associated with the Givens
      rotation.


   s
      Buffer holding the parameter ``s`` associated with the Givens
      rotation.


rotg (USM Version)
------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. container:: dlsyntaxpara


         .. cpp:function::  sycl::event onemkl::blas::rotg(sycl::queue &queue, T *a, T *b, T_real *c, T *s, const sycl::vector_class<sycl::event> &dependencies = {})
   .. container:: section


      .. rubric:: Input Parameters
         :class: sectiontitle


      queue
         The queue where the routine should be executed


      a
         Pointer to the ``x``-coordinate of the point.


      b
         Pointer to the ``y``-coordinate of the point.


      dependencies
         List of events to wait for before starting computation, if any.
         If omitted, defaults to no dependencies.


   .. container:: section


      .. rubric:: Output Parameters
         :class: sectiontitle


      a
         Pointer to the parameter ``r`` associated with the Givens
         rotation.


      b
         Pointer to the parameter ``z`` associated with the Givens
         rotation.


      c
         Pointer to the parameter ``c`` associated with the Givens
         rotation.


      s
         Pointer to the parameter ``s`` associated with the Givens
         rotation.


   .. container:: section


      .. rubric:: Return Values
         :class: sectiontitle


      Output event to wait on to ensure computation is complete.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-1-routines`
