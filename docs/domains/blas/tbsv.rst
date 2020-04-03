.. _tbsv:

tbsv
====


.. container::


   Solves a system of linear equations whose coefficients are in a
   triangular band matrix.


   .. container:: section
      :name: GUID-4AC7186F-2D61-44C2-95BC-5981E750A021


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void tbsv(queue &exec_queue, uplo upper_lower,      transpose trans, diag unit_nonunit, std::int64_t n, std::int64_t      k, buffer<T,1> &a, std::int64_t lda, buffer<T,1> &x, std::int64_t      incx)

      ``tbsv`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``float`` 
         * -  ``double`` 
         * -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 




.. container:: section
   :name: GUID-5AF4221C-AB14-4F9B-97A8-CAA78DF05E36


   .. rubric:: Description
      :name: description
      :class: sectiontitle


   The tbsv routines solve a system of linear equations whose
   coefficients are in a triangular band matrix. The operation is
   defined as


  


      op(A)*x = b


   where:


   op(``A``) is one of op(``A``) = ``A``, or op(``A``) =
   ``A``\ :sup:`T`, or op(``A``) = ``A``\ :sup:`H`,


   ``A`` is an ``n``-by-``n`` unit or non-unit, upper or lower
   triangular band matrix, with (``k`` + 1) diagonals,


   ``b`` and ``x`` are vectors of length ``n``.


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
      Number of rows and columns of ``A``. Must be at least zero.


   k
      Number of sub/super-diagonals of the matrix ``A``. Must be at
      least zero.


   a
      Buffer holding input matrix ``A``. Must have size at least
      ``lda``\ \*\ ``n``. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   lda
      Leading dimension of matrix ``A``. Must be at least (``k`` + 1),
      and positive.


   x
      Buffer holding input vector ``x``. The buffer must be of size at
      least (1 + (``n`` - 1)*abs(``incx``)). See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   incx
      Stride of vector ``x``.


.. container:: section
   :name: GUID-24B3C6B8-7FBD-4B24-84F2-242635B3026E


   .. rubric:: Output Parameters
      :name: output-parameters
      :class: sectiontitle


   x
      Buffer holding the solution vector ``x``.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-2-routines`
      


.. container::

