.. _trmv:

trmv
====


.. container::


   Computes a matrix-vector product using a triangular matrix.


   .. container:: section
      :name: GUID-15041079-C2F5-4D3C-85C2-262E184F7FFE


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void trmv(queue &exec_queue, uplo upper_lower,      transpose trans, diag unit_nonunit, std::int64_t n, buffer<T,1>      &a, std::int64_t lda, buffer<T,1> &x, std::int64_t incx)

      ``trmv`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``float`` 
         * -  ``double`` 
         * -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 




.. container:: section
   :name: GUID-420DC613-E11B-48A8-B73F-55B55EBFC3B7


   .. rubric:: Description
      :name: description
      :class: sectiontitle


   The trmv routines compute a matrix-vector product with a triangular
   matrix. The operation is defined


  


      x <- op(A)*x


   where:


   op(``A``) is one of op(``A``) = ``A``, or op(``A``) =
   ``A``\ :sup:`T`, or op(``A``) = ``A``\ :sup:`H`,


   ``A`` is an ``n``-by-``n`` unit or non-unit, upper or lower
   triangular band matrix,


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
      ``lda``\ \*\ ``n``. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   lda
      Leading dimension of matrix ``A``. Must be at least ``n``, and
      positive.


   x
      Buffer holding input vector ``x``. The buffer must be of size at
      least (1 + (``n`` - 1)*abs(``incx``)). See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   incx
      Stride of vector ``x``.


.. container:: section
   :name: GUID-7BF1D5C9-EB8C-4BD6-B0E7-A66DAC3221F9


   .. rubric:: Output Parameters
      :name: output-parameters
      :class: sectiontitle


   x
      Buffer holding the updated vector ``x``.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-2-routines`
      


.. container::

