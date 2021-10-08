.. _onemkl_blas_rotg:

rotg
====

Computes the parameters for a Givens rotation.

.. _onemkl_blas_rotg_description:

.. rubric:: Description

Given the Cartesian coordinates ``(a, b)`` of a point, the ``rotg``
routines return the parameters ``c``, ``s``, ``r``, and ``z``
associated with the Givens rotation. The parameters ``c`` and ``s``
define a unitary matrix such that:

.. math::
      
      \begin{bmatrix}c & s \\ -s & c\end{bmatrix}.
      \begin{bmatrix}a \\ b\end{bmatrix}
      =\begin{bmatrix}r \\ 0\end{bmatrix} 

The parameter ``z`` is defined such that if \|\ ``a``\ \| >
\|\ ``b``\ \|, ``z`` is ``s``; otherwise if ``c`` is not 0 ``z`` is
1/``c``; otherwise ``z`` is 1.

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

.. _onemkl_blas_rotg_buffer:

rotg (Buffer Version)
---------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       void rotg(sycl::queue &queue,
                 sycl::buffer<T,1> &a,
                 sycl::buffer<T,1> &b,
                 sycl::buffer<T_real,1> &c,
                 sycl::buffer<T,1> &s)
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       void rotg(sycl::queue &queue,
                 sycl::buffer<T,1> &a,
                 sycl::buffer<T,1> &b,
                 sycl::buffer<T_real,1> &c,
                 sycl::buffer<T,1> &s)
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed

   a
      Buffer holding the ``x``-coordinate of the point.

   b
      Buffer holding the ``y``-coordinate of the point.

.. container:: section

   .. rubric:: Output Parameters

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


.. _onemkl_blas_rotg_usm:

rotg (USM Version)
------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       sycl::event rotg(sycl::queue &queue,
                        T *a,
                        T *b,
                        T_real *c,
                        T *s,
                        const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       sycl::event rotg(sycl::queue &queue,
                        T *a,
                        T *b,
                        T_real *c,
                        T *s,
                        const std::vector<sycl::event> &dependencies = {})
   }

.. container:: section

   .. rubric:: Input Parameters

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

   Output event to wait on to ensure computation is complete.


   **Parent topic:** :ref:`blas-level-1-routines`
