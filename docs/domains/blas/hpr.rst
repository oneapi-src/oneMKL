.. _hpr:

hpr
===


.. container::


   Computes a rank-1 update of a Hermitian packed matrix.


   .. container:: section
      :name: GUID-61DC4DBA-9357-4129-B8A3-931E2E7335D4


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void hpr(queue &exec_queue, uplo upper_lower,      std::int64_t n, T alpha, buffer<T,1> &x, std::int64_t incx,      buffer<T,1> &a)

      ``hpr`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 




.. container:: section
   :name: GUID-02B8128C-02CE-4D5C-BE5D-DFD088C90475


   .. rubric:: Description
      :name: description
      :class: sectiontitle


   The hpr routines compute a scalar-vector-vector product and add the
   result to a Hermitian packed matrix. The operation is defined as


  


      A <- alpha*x*x :sup:`H` + A


   where:


   ``alpha`` is scalar,


   ``A`` is an ``n``-by-``n`` Hermitian matrix, supplied in packed form,


   ``x`` is a vector of length ``n``.


.. container:: section
   :name: GUID-E1436726-01FE-4206-871E-B905F59A96B4


   .. rubric:: Input Parameters
      :name: input-parameters
      :class: sectiontitle


   exec_queue
      The queue where the routine should be executed.


   upper_lower
      Specifies whether ``A`` is upper or lower triangular. See
      :ref:`onemkl_datatypes` for more
      details.


   n
      Number of rows and columns of ``A``. Must be at least zero.


   alpha
      Scaling factor for the matrix-vector product.


   x
      Buffer holding input vector ``x``. The buffer must be of size at
      least (1 + (``n`` - 1)*abs(``incx``)). See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   incx
      Stride of vector ``x``.


   a
      Buffer holding input matrix ``A``. Must have size at least
      (``n``\ \*(``n``-1))/2. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


      The imaginary part of the diagonal elements need not be set and
      are assumed to be zero


.. container:: section
   :name: GUID-7261182A-450B-46F5-8C61-7133597D3530


   .. rubric:: Output Parameters
      :name: output-parameters
      :class: sectiontitle


   a
      Buffer holding the updated upper triangularpart of the Hermitian
      matrix ``A`` if ``upper_lower =upper``, or the updated lower
      triangular part of theHermitian matrix ``A`` if
      ``upper_lower =lower``.


      The imaginary parts of the diagonal elements are set tozero.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-2-routines`
      


.. container::

