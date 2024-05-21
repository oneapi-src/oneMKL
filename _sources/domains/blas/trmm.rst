.. _onemkl_blas_trmm:

trmm
====

Computes a matrix-matrix product where one input matrix is triangular
and one input matrix is general.

.. _onemkl_blas_trmm_description:

.. rubric:: Description

The ``trmm`` routines compute a scalar-matrix-matrix product where one of
the matrices in the multiplication is triangular. The argument
``left_right`` determines if the triangular matrix, ``A``, is on the
left of the multiplication (``left_right`` = ``side::left``) or on
the right (``left_right`` = ``side::right``). Depending on
``left_right``. The operation is defined as:

.. math::

      B \leftarrow alpha*op(A)*B

or

.. math::

      B \leftarrow alpha*B*op(A)

where:

op(``A``) is one of op(``A``) = *A*, or op(``A``) = ``A``\ :sup:`T`,
or op(``A``) = ``A``\ :sup:`H`,

``alpha`` is a scalar,

``A`` is a triangular matrix, and ``B`` is a general matrix.

Here ``B`` is ``m`` x ``n`` and ``A`` is either ``m`` x ``m`` or
``n`` x ``n``, depending on ``left_right``.

``trmm`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. _onemkl_blas_trmm_buffer:

trmm (Buffer Version)
---------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       void trmm(sycl::queue &queue,
                 onemkl::uplo upper_lower,
                 onemkl::transpose transa,
                 onemkl::diag unit_diag,
                 std::int64_t m,
                 std::int64_t n,
                 T alpha,
                 sycl::buffer<T,1> &a,
                 std::int64_t lda,
                 sycl::buffer<T,1> &b,
                 std::int64_t ldb)
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       void trmm(sycl::queue &queue,
                 onemkl::uplo upper_lower,
                 onemkl::transpose transa,
                 onemkl::diag unit_diag,
                 std::int64_t m,
                 std::int64_t n,
                 T alpha,
                 sycl::buffer<T,1> &a,
                 std::int64_t lda,
                 sycl::buffer<T,1> &b,
                 std::int64_t ldb)
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   left_right
      Specifies whether ``A`` is on the left side of the multiplication
      (``side::left``) or on the right side (``side::right``). See :ref:`onemkl_datatypes` for more details.

   uplo
      Specifies whether the matrix ``A`` is upper or lower triangular. See :ref:`onemkl_datatypes` for more details.

   trans
      Specifies op(``A``), the transposition operation applied to ``A``. See :ref:`onemkl_datatypes` for more details.

   unit_diag
      Specifies whether ``A`` is assumed to be unit triangular (all
      diagonal elements are 1). See :ref:`onemkl_datatypes` for more details.

   m
      Specifies the number of rows of ``B``. The value of ``m`` must be
      at least zero.

   n
      Specifies the number of columns of ``B``. The value of ``n`` must
      be at least zero.

   alpha
      Scaling factor for the matrix-matrix product.

   a
      Buffer holding input matrix ``A``. Must have size at least
      ``lda``\ \*\ ``m`` if ``left_right`` = ``side::left``, or
      ``lda``\ \*\ ``n`` if ``left_right`` = ``side::right``. See
      :ref:`matrix-storage` for
      more details.

   lda
      Leading dimension of ``A``. Must be at least ``m`` if
      ``left_right`` = ``side::left``, and at least ``n`` if
      ``left_right`` = ``side::right``. Must be positive.

   b
      Buffer holding input/output matrix ``B``. Must have size at
      least ``ldb``\ \*\ ``n`` if column major layout is used to store
      matrices or at least ``ldb``\ \*\ ``m`` if row major layout is
      used to store matrices. See :ref:`matrix-storage` for more details.

   ldb
      Leading dimension of ``B``. It must be positive and at least
      ``m`` if column major layout is used to store matrices or at
      least ``n`` if row major layout is used to store matrices.

.. container:: section

   .. rubric:: Output Parameters

   b
      Output buffer, overwritten by ``alpha``\ \*op(``A``)\*\ ``B`` or
      ``alpha``\ \*\ ``B``\ \*op(``A``).

.. container:: section

   .. rubric:: Notes

   If ``alpha`` = 0, matrix ``B`` is set to zero, and ``A`` and ``B`` do
   not need to be initialized at entry.

      

.. _onemkl_blas_trmm_usm:

trmm (USM Version)
------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       sycl::event trmm(sycl::queue &queue,
                        onemkl::uplo upper_lower,
                        onemkl::transpose transa,
                        onemkl::diag unit_diag,
                        std::int64_t m,
                        std::int64_t n,
                        T alpha,
                        const T* a,
                        std::int64_t lda,
                        T* b,
                        std::int64_t ldb,
                        const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       sycl::event trmm(sycl::queue &queue,
                        onemkl::uplo upper_lower,
                        onemkl::transpose transa,
                        onemkl::diag unit_diag,
                        std::int64_t m,
                        std::int64_t n,
                        T alpha,
                        const T* a,
                        std::int64_t lda,
                        T* b,
                        std::int64_t ldb,
                        const std::vector<sycl::event> &dependencies = {})
   }

.. container:: section
   
   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   left_right
      Specifies whether ``A`` is on the left side of the
      multiplication (``side::left``) or on the right side
      (``side::right``). See :ref:`onemkl_datatypes` for more details.

   uplo
      Specifies whether the matrix ``A`` is upper or lower
      triangular. See :ref:`onemkl_datatypes` for more details.

   trans
      Specifies op(``A``), the transposition operation applied to
      ``A``. See :ref:`onemkl_datatypes` for more details.

   unit_diag
      Specifies whether ``A`` is assumed to be unit triangular (all
      diagonal elements are 1). See :ref:`onemkl_datatypes` for more details.

   m
      Specifies the number of rows of ``B``. The value of ``m`` must
      be at least zero.

   n
      Specifies the number of columns of ``B``. The value of ``n``
      must be at least zero.

   alpha
      Scaling factor for the matrix-matrix product.

   a
      Pointer to input matrix ``A``. Must have size at least
      ``lda``\ \*\ ``m`` if ``left_right`` = ``side::left``, or
      ``lda``\ \*\ ``n`` if ``left_right`` = ``side::right``. See
      :ref:`matrix-storage` for
      more details.

   lda
      Leading dimension of ``A``. Must be at least ``m`` if
      ``left_right`` = ``side::left``, and at least ``n`` if
      ``left_right`` = ``side::right``. Must be positive.

   b
      Pointer to input/output matrix ``B``. Must have size at
      least ``ldb``\ \*\ ``n`` if column major layout is used to store
      matrices or at least ``ldb``\ \*\ ``m`` if row major layout is
      used to store matrices. See :ref:`matrix-storage` for more details.

   ldb
      Leading dimension of ``B``. It must be positive and at least
      ``m`` if column major layout is used to store matrices or at
      least ``n`` if row major layout is used to store matrices.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   b
      Pointer to the output matrix, overwritten by
      ``alpha``\ \*op(``A``)\*\ ``B`` or
      ``alpha``\ \*\ ``B``\ \*op(``A``).

.. container:: section

   .. rubric:: Notes

   If ``alpha`` = 0, matrix ``B`` is set to zero, and ``A`` and ``B``
   do not need to be initialized at entry.

.. container:: section

   .. rubric:: Return Values

   Output event to wait on to ensure computation is complete.


   **Parent topic:** :ref:`blas-level-3-routines`
