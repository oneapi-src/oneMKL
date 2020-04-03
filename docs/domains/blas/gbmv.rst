.. _gbmv:

gbmv
====


.. container::


   Computes a matrix-vector product with a general band matrix.


   .. container:: section
      :name: GUID-870EA7B0-09B5-43FF-90A4-6378B5D94B55


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void gbmv(queue &exec_queue, transpose trans,      std::int64_t m, std::int64_t n, std::int64_t kl, std::int64_t ku,      T alpha, buffer<T,1> &a, std::int64_t lda, buffer<T,1> &x,      std::int64_t incx, T beta, buffer<T,1> &y, std::int64_t incy)

      ``gbmv`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``float`` 
         * -  ``double`` 
         * -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 




.. container:: section
   :name: GUID-71614419-BC91-4A1A-B743-FE52767C4926


   .. rubric:: Description
      :name: description
      :class: sectiontitle


   The ``gbmv`` routines compute a scalar-matrix-vector product and add
   the result to a scalar-vector product, with a general band matrix.
   The operation is defined as


  


      y <- alpha*op(A)*x + beta*y


   where:


   -  op(``A``) is one of op(``A``) = ``A``, or op(``A``) =
      ``A``\ :sup:`T`, or op(``A``) = ``A``\ :sup:`H`,


   -  ``alpha`` and ``beta`` are scalars,


   -  ``A`` is an ``m``-by-``n`` matrix with ``kl`` sub-diagonals and
      ``ku`` super-diagonals,


   -  ``x`` and ``y`` are vectors.


.. container:: section
   :name: GUID-E1436726-01FE-4206-871E-B905F59A96B4


   .. rubric:: Input Parameters
      :name: input-parameters
      :class: sectiontitle


   exec_queue
      The queue where the routine should be executed.


   trans
      Specifies op(``A``), the transposition operation applied to ``A``.
      See
      :ref:`onemkl_datatypes` for more
      details.


   m
      Number of rows of ``A``. Must be at least zero.


   n
      Number of columns of ``A``. Must be at least zero.


   kl
      Number of sub-diagonals of the matrix ``A``. Must be at least
      zero.


   ku
      Number of super-diagonals of the matrix ``A``. Must be at least
      zero.


   alpha
      Scaling factor for the matrix-vector product.


   a
      Buffer holding input matrix ``A``. Must have size at least
      ``lda``\ \*\ ``n``. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   lda
      Leading dimension of matrix ``A``. Must be at least (``kl`` +
      ``ku`` + 1), and positive.


   x
      Buffer holding input vector ``x``. The length ``len`` of vector
      ``x`` is ``n`` if ``A`` is not transposed, and ``m`` if ``A`` is
      transposed. The buffer must be of size at least (1 + (``len`` -
      1)*abs(``incx``)). See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   incx
      Stride of vector ``x``.


   beta
      Scaling factor for vector ``y``.


   y
      Buffer holding input/output vector ``y``. The length ``len`` of
      vector ``y`` is ``m``, if ``A`` is not transposed, and ``n`` if
      ``A`` is transposed. The buffer must be of size at least (1 +
      (``len`` - 1)*abs(``incy``)) where ``len`` is this length. See
      `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   incy
      Stride of vector ``y``.


.. container:: section
   :name: GUID-4B31584D-BC63-4032-A4A7-61BF3F163165


   .. rubric:: Output Parameters
      :name: output-parameters
      :class: sectiontitle


   y
      Buffer holding the updated vector ``y``.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-2-routines`
      


.. container::

