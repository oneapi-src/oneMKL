.. _gemmt:

gemmt
=====


.. container::


   Computes a matrix-matrix product with general matrices, but updates
   only the upper or lower triangular part of the result matrix.


   .. container:: section
      :name: GUID-7885D940-FAC1-4F37-9E1C-A022DED99EBD


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. container:: dlsyntaxpara


         .. cpp:function::  void gemmt(queue &exec_queue, uplo         upper_lower, transpose transa, transpose transb, std::int64_t         n, std::int64_t k, T alpha, buffer<T,1> &a, std::int64_t lda,         buffer<T,1> &b, std::int64_t ldb, T beta, buffer<T,1> &c,         std::int64_t ldc)

         ``gemmt`` supports the following precisions.


         .. list-table:: 
            :header-rows: 1

            * -  T 
            * -  ``float`` 
            * -  ``double`` 
            * -  ``std::complex<float>`` 
            * -  ``std::complex<double>`` 




   .. container:: section
      :name: GUID-14237C95-6322-47A4-BC11-D3CDD2118C42


      .. rubric:: Description
         :name: description
         :class: sectiontitle


      The gemmt routines compute a scalar-matrix-matrix product and add
      the result to the upper or lower part of a scalar-matrix product,
      with general matrices. The operation is defined as:


      ::


         C <- alpha*op(A)*op(B) + beta*C 


      where:


      -  op(X) is one of op(X) = X, or op(X) = X\ :sup:`T`, or op(X) =
         X\ :sup:`H`


      -  ``alpha`` and ``beta`` are scalars


      -  ``A``, ``B``, and ``C`` are matrices


      Here, op(``A``) is ``n`` x ``k``, op(``B``) is ``k`` x ``n``, and
      ``C`` is ``n`` x ``n``.


   .. container:: section
      :name: GUID-863264A0-4CE9-495F-A617-102E46D7A41A


      .. rubric:: Input Parameters
         :name: input-parameters
         :class: sectiontitle


      exec_queue
         The queue where the routine should be executed.


      upper_lower
         Specifies whether ``C``\ ’s data is stored in its upper or
         lower triangle. See
         :ref:`onemkl_datatypes` for
         more details.


      transa
         Specifies op(``A``), the transposition operation applied to
         ``A``. See
         :ref:`onemkl_datatypes` for
         more details.


      transb
         Specifies op(``B``), the transposition operation applied to
         ``B``. See
         :ref:`onemkl_datatypes` for
         more details.


      n
         Number of columns of op(``A``), columns of op(``B``), and
         columns of\ ``C``. Must be at least zero.


      k
         Number of columns of op(``A``) and rows of op(``B``). Must be
         at least zero.


      alpha
         Scaling factor for the matrix-matrix product.


      a
         Buffer holding the input matrix ``A``.


         If ``A`` is not transposed, ``A`` is an ``n``-by-``k`` matrix
         so the array ``a`` must have size at least ``lda``\ \*\ ``k``.


         If ``A`` is transposed, ``A`` is a ``k``-by-``n`` matrix so the
         array ``a`` must have size at least ``lda``\ \*\ ``n``.


         See `Matrix
         Storage <../matrix-storage.html>`__ for
         more details.


      lda
         Leading dimension of ``A``. Must be at least ``n`` if ``A`` is
         not transposed, and at least ``k`` if ``A`` is transposed. Must
         be positive.


      b
         Buffer holding the input matrix ``B``.


         If ``B`` is not transposed, ``B`` is a ``k``-by-``n`` matrix so
         the array ``b`` must have size at least ``ldb``\ \*\ ``n``.


         If ``B`` is transposed, ``B`` is an ``n``-by-``k`` matrix so
         the array ``b`` must have size at least ``ldb``\ \*\ ``k``.


         See `Matrix
         Storage <../matrix-storage.html>`__ for
         more details.


      ldb
         Leading dimension of ``B``. Must be at least ``k`` if ``B`` is
         not transposed, and at least ``n`` if ``B`` is transposed. Must
         be positive.


      beta
         Scaling factor for matrix ``C``.


      c
         Buffer holding the input/output matrix ``C``. Must have size at
         least ``ldc`` \* ``n``. See `Matrix
         Storage <../matrix-storage.html>`__ for
         more details.


      ldc
         Leading dimension of ``C``. Must be positive and at least
         ``m``.


   .. container:: section
      :name: GUID-1E4953E6-F7B1-4FEE-BA5A-8C4BD51DC700


      .. rubric:: Output Parameters
         :name: output-parameters
         :class: sectiontitle


      c
         Output buffer, overwritten by the upper or lower triangular
         part ofalpha\*op(``A``)*op(``B``) + beta\*\ ``C``.


   .. container:: section
      :name: GUID-AC72653A-4AC8-4B9D-B7A9-13A725AA19BF


      .. rubric:: Notes
         :name: notes
         :class: sectiontitle


      If ``beta`` = 0, matrix ``C`` does not need to be initialized
      before calling gemmt.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-like-extensions`
      


.. container::

