.. _rotg:

rotg
====


.. container::


   Computes the parameters for a Givens rotation.


   .. container:: section
      :name: GUID-E4B6E693-AC8C-4BB3-A197-3EB9E905B925


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void rotg(queue &exec_queue, buffer<T,1> &a,      buffer<T,1> &b, buffer<T_real,1> &c, buffer<T,1> &s)

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
   :name: GUID-5614B81D-C736-4714-88AB-29B38F9B3589


   .. rubric:: Description
      :name: description
      :class: sectiontitle


   Given the Cartesian coordinates ``(a, b)`` of a point, the rotg
   routines return the parameters ``c``, ``s``, ``r``, and ``z``
   associated with the Givens rotation. The parameters ``c`` and ``s``
   define a unitary matrix such that:


   The parameter ``z`` is defined such that if \|\ ``a``\ \| >
   \|\ ``b``\ \|, ``z`` is ``s``; otherwise if ``c`` is not 0 ``z`` is
   1/``c``; otherwise ``z`` is 1.


.. container:: section
   :name: GUID-C2003328-15AA-4DF0-A417-40BECCA7DEA3


   .. rubric:: Input Parameters
      :name: input-parameters
      :class: sectiontitle


   exec_queue
      The queue where the routine should be executed


   a
      Buffer holding the ``x``-coordinate of the point.


   b
      Buffer holding the ``y``-coordinate of the point.


.. container:: section
   :name: GUID-3B7937E3-2DF7-49A3-8F1E-2C9406BB4E88


   .. rubric:: Output Parameters
      :name: output-parameters
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


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-1-routines`
      


.. container::

