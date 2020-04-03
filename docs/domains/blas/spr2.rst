.. _spr2:

spr2
====


.. container::


   Computes a rank-2 update of a symmetric packed matrix.


   .. container:: section
      :name: GUID-44B72132-1EC0-41FA-9189-4596CFD651B0


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void spr2(queue &exec_queue, uplo upper_lower,      std::int64_t n, T alpha, buffer<T,1> &x, std::int64_t incx,      buffer<T,1> &y, std::int64_t incy, buffer<T,1> &a)

      ``spr`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``float`` 
         * -  ``double`` 




.. container:: section
   :name: GUID-3AF7EB4D-B3FE-4C0A-B7A0-6E286D4C642F


   .. rubric:: Description
      :name: description
      :class: sectiontitle


   The spr2 routines compute two scalar-vector-vector products and add
   them to a symmetric packed matrix. The operation is defined as


  


      A <- alpha*x*y :sup:`T` + alpha*y*x :sup:`T` + A


   where:


   ``alpha`` is scalar,


   ``A`` is an ``n``-by-``n`` symmetric matrix, supplied in packed form,


   ``x`` and ``y`` are vectors of length ``n``.


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


   y
      Buffer holding input/output vector ``y``. The buffer must be of
      size at least (1 + (``n`` - 1)*abs(``incy``)). See `Matrix and
      Vector Storage <../matrix-storage.html>`__
      for more details.


   incy
      Stride of vector ``y``.


   a
      Buffer holding input matrix ``A``. Must have size at least
      (``n``\ \*(``n``-1))/2. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


.. container:: section
   :name: GUID-9796BA93-31FB-40B9-B139-219905913736


   .. rubric:: Output Parameters
      :name: output-parameters
      :class: sectiontitle


   **sycl:**
       



   a
      Buffer holding the updated upper triangularpart of the symmetric
      matrix ``A`` if ``upper_lower =upper`` or the updated lower
      triangular part of thesymmetric matrix ``A`` if
      ``upper_lower =lower``.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-2-routines`
      


.. container::

