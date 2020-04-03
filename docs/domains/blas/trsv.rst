.. _trsv:

trsv
====


.. container::


   Solves a system of linear equations whose coefficients are in a
   triangular matrix.


   .. container:: section
      :name: GUID-9BA4C1B6-479B-41B1-BCA8-7826F40DA952


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void trsv(queue &exec_queue, uplo upper_lower,      transpose trans, diag unit_nonunit, std::int64_t n, std::int64_t      k, buffer<T,1> &a, std::int64_t lda, buffer<T,1> &x, std::int64_t      incx)

      ``trsv`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``float`` 
         * -  ``double`` 
         * -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 




.. container:: section
   :name: GUID-D500B67B-5DD6-4471-B0BD-53FD9A3C7BF2


   .. rubric:: Description
      :name: description
      :class: sectiontitle


   The trsv routines compute a matrix-vector product with a triangular
   band matrix. The operation is defined as


  


      op(A)*x = b


   where:


   op(``A``) is one of op(``A``) = ``A``, or op(``A``) =
   ``A``\ :sup:`T`, or op(``A``) = ``A``\ :sup:`H`,


   ``A`` is an ``n``-by-``n`` unit or non-unit, upper or lower
   triangular matrix,


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
      Buffer holding the ``n``-element right-hand side vector ``b``. The
      buffer must be of size at least (1 + (``n`` - 1)*abs(``incx``)).
      See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   incx
      Stride of vector ``x``.


.. container:: section
   :name: GUID-7E0AF44F-2D83-41A3-A58E-50400ECDBD9A


   .. rubric:: Output Parameters
      :name: output-parameters
      :class: sectiontitle


   x
      Buffer holding the solution vector ``x``.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-2-routines`
      


.. container::

