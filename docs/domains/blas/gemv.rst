.. _gemv:

gemv
====


.. container::


   Computes a matrix-vector product using a general matrix.


   .. container:: section
      :name: GUID-EA8D6705-E7C2-42E2-BE80-D9AD83645FCC


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void gemv(queue &exec_queue, transpose trans,      std::int64_t m, std::int64_t n, T alpha, buffer<T,1> &a,      std::int64_t lda, buffer<T,1> &x, std::int64_t incx, T beta,      buffer<T,1> &y, std::int64_t incy)

      gemv supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``float`` 
         * -  ``double`` 
         * -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 




.. container:: section
   :name: GUID-AE220EED-6066-4881-8B3C-35207BAB0105


   .. rubric:: Description
      :name: description
      :class: sectiontitle


   The gemv routines compute a scalar-matrix-vector product and add the
   result to a scalar-vector product, with a general matrix. The
   operation is defined as


  


      y  <- alpha*op(A)*x + beta*y


   where:


   op(``A``) is one of op(``A``) = ``A``, or op(``A``) =
   ``A``\ :sup:`T`, or op(``A``) = ``A``\ :sup:`H`,


   ``alpha`` and ``beta`` are scalars,


   ``A`` is an ``m``-by-``n`` matrix, and ``x``, ``y`` are vectors.


.. container:: section
   :name: GUID-F3E8F201-6033-45A1-A326-CA4CFB631C3A


   .. rubric:: Input Parameters
      :name: input-parameters
      :class: sectiontitle


   exec_queue
      The queue where the routine should be executed.


   trans
      Specifies ``op(A)``, the transposition operation applied to ``A``.
      See
      :ref:`onemkl_datatypes` for more
      details.


   m
      Specifies the number of rows of the matrix ``A``. The value of
      ``m`` must be at least zero.


   n
      Specifies the number of columns of the matrix ``A``. The value of
      ``n`` must be at least zero.


   alpha
      Scaling factor for the matrix-vector product.


   a
      The buffer holding the input matrix ``A``. Must have a size of at
      least ``lda``\ \*n. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   lda
      The leading dimension of matrix ``A``. It must be at least m, and
      positive.


   x
      Buffer holding input vector ``x``. The length ``len`` of vector
      ``x`` is ``n`` if ``A`` is not transposed, and ``m`` if ``A`` is
      transposed. The buffer must be of size at least (1 + (``len`` -
      1)*abs(``incx``)). See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   incx
      The stride of vector ``x``.


   beta
      The scaling factor for vector ``y``.


   y
      Buffer holding input/output vector ``y``. The length ``len`` of
      vector ``y`` is ``m``, if ``A`` is not transposed, and ``n`` if
      ``A`` is transposed. The buffer must be of size at least (1 +
      (``len`` - 1)*abs(``incy``)) where ``len`` is this length. See
      `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   incy
      The stride of vector ``y``.


.. container:: section
   :name: GUID-1533BCA6-E652-4A08-A82D-162F3CEBDD29


   .. rubric:: Output Parameters
      :name: output-parameters
      :class: sectiontitle


   y
      The buffer holding updated vector ``y``.



.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-2-routines`
      


.. container::

