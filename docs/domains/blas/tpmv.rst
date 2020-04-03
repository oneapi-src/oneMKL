.. _tpmv:

tpmv
====


.. container::


   Computes a matrix-vector product using a triangular packed matrix.


   .. container:: section
      :name: GUID-5785B6D6-DB9C-43FA-B98A-009D5E077A9D


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void tpmv(queue &exec_queue, uplo upper_lower,      transpose trans, diag unit_nonunit, std::int64_t n, buffer<T,1>      &a, buffer<T,1> &x, std::int64_t incx)

      ``tpmv`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``float`` 
         * -  ``double`` 
         * -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 




.. container:: section
   :name: GUID-A045480A-2EC1-4C73-A836-468324FCC85A


   .. rubric:: Description
      :name: description
      :class: sectiontitle


   The tpmv routines compute a matrix-vector product with a triangular
   packed matrix. The operation is defined as


  


      x <- op(A)*x


   where:


   op(``A``) is one of op(``A``) = ``A``, or op(``A``) =
   ``A``\ :sup:`T`, or op(``A``) = ``A``\ :sup:`H`,


   ``A`` is an ``n``-by-``n`` unit or non-unit, upper or lower
   triangular band matrix, supplied in packed form,


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


   trans
      Specifies op(``A``), the transposition operation applied to ``A``.
      See
      :ref:`onemkl_datatypes` for more
      details.


   unit_nonunit
      Specifies whether the matrix ``A`` is unit triangular or not. See
      :ref:`onemkl_datatypes`
      for more details.


   n
      Numbers of rows and columns of ``A``. Must be at least zero.


   a
      Buffer holding input matrix ``A``. Must have size at least
      (``n``\ \*(``n``\ +1))/2. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   x
      Buffer holding input vector ``x``. The buffer must be of size at
      least (1 + (``n`` - 1)*abs(``incx``)). See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   incx
      Stride of vector ``x``.


.. container:: section
   :name: GUID-180038D9-902F-4B20-AB6B-E38F2A6C83E4


   .. rubric:: Output Parameters
      :name: output-parameters
      :class: sectiontitle


   x
      Buffer holding the updated vector ``x``.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-2-routines`
      


.. container::

