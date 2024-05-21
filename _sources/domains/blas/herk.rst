.. _onemkl_blas_herk:

herk
====

Performs a Hermitian rank-k update.

.. _onemkl_blas_herk_description:

.. rubric:: Description

The ``herk`` routines compute a rank-k update of a Hermitian matrix
``C`` by a general matrix ``A``. The operation is defined as:

.. math::

      C \leftarrow alpha*op(A)*op(A)^H + beta*C

where:

op(``X``) is one of op(``X``) = ``X`` or op(``X``) = ``X``\ :sup:`H`,

``alpha`` and ``beta`` are real scalars,

``C`` is a Hermitian matrix and ``A`` is a general matrix.

Here op(``A``) is ``n`` x ``k``, and ``C`` is ``n`` x ``n``.

``herk`` supports the following precisions:

   .. list-table:: 
      :header-rows: 1

      * -  T 
        -  T_real 
      * -  ``std::complex<float>`` 
        -  ``float`` 
      * -  ``std::complex<double>`` 
        -  ``double`` 

.. _onemkl_blas_herk_buffer:

herk (Buffer Version)
---------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       void herk(sycl::queue &queue,
                 onemkl::uplo upper_lower,
                 onemkl::transpose trans,
                 std::int64_t n,
                 std::int64_t k,
                 T_real alpha,
                 sycl::buffer<T,1> &a,
                 std::int64_t lda,
                 T_real beta,
                 sycl::buffer<T,1> &c,
                 std::int64_t ldc)
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       void herk(sycl::queue &queue,
                 onemkl::uplo upper_lower,
                 onemkl::transpose trans,
                 std::int64_t n,
                 std::int64_t k,
                 T_real alpha,
                 sycl::buffer<T,1> &a,
                 std::int64_t lda,
                 T_real beta,
                 sycl::buffer<T,1> &c,
                 std::int64_t ldc)
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   upper_lower
      Specifies whether ``A``'s data is stored in its upper or lower
      triangle. See :ref:`onemkl_datatypes` for more details.

   trans
      Specifies op(``A``), the transposition operation applied to ``A``. See
      :ref:`onemkl_datatypes` for more
      details. Supported operations are ``transpose::nontrans`` and
      ``transpose::conjtrans``.

   n
      The number of rows and columns in ``C``.The value of ``n`` must be
      at least zero.

   k
      Number of columns in op(``A``).

      The value of ``k`` must be at least zero.

   alpha
      Real scaling factor for the rank-k update.

   a
      Buffer holding input matrix ``A``.

      .. list-table::
         :header-rows: 1

         * -
           - ``trans`` = ``transpose::nontrans``
           - ``trans`` = ``transpose::trans`` or ``transpose::conjtrans``
         * - Column major
           - ``A`` is an ``n``-by-``k`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``k``.
           - ``A`` is an ``k``-by-``n`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``n``
         * - Row major
           - ``A`` is an ``n``-by-``k`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``n``.
           - ``A`` is an ``k``-by-``n`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``k``.

      See :ref:`matrix-storage` for
      more details.

   lda
      The leading dimension of ``A``. It must be positive.

      .. list-table::
         :header-rows: 1

         * -
           - ``trans`` = ``transpose::nontrans``
           - ``trans`` = ``transpose::trans`` or ``transpose::conjtrans``
         * - Column major
           - ``lda`` must be at least ``n``.
           - ``lda`` must be at least ``k``.
         * - Row major
           - ``lda`` must be at least ``k``.
           - ``lda`` must be at least ``n``.

   beta
      Real scaling factor for matrix ``C``.

   c
      Buffer holding input/output matrix ``C``. Must have size at least
      ``ldc``\ \*\ ``n``. See :ref:`matrix-storage` for
      more details.

   ldc
      Leading dimension of ``C``. Must be positive and at least ``n``.

.. container:: section

   .. rubric:: Output Parameters

   c
      The output buffer, overwritten by
      ``alpha``\ \*op(``A``)*op(``A``)\ :sup:`T` + ``beta``\ \*\ ``C``.
      The imaginary parts of the diagonal elements are set to zero.

      

.. _onemkl_blas_herk_usm:

herk (USM Version)
------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       sycl::event herk(sycl::queue &queue,
                        onemkl::uplo upper_lower,
                        onemkl::transpose trans,
                        std::int64_t n,
                        std::int64_t k,
                        T_real alpha,
                        const T* a,
                        std::int64_t lda,
                        T_real beta,
                        T* c,
                        std::int64_t ldc,
                        const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       sycl::event herk(sycl::queue &queue,
                        onemkl::uplo upper_lower,
                        onemkl::transpose trans,
                        std::int64_t n,
                        std::int64_t k,
                        T_real alpha,
                        const T* a,
                        std::int64_t lda,
                        T_real beta,
                        T* c,
                        std::int64_t ldc,
                        const std::vector<sycl::event> &dependencies = {})
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   upper_lower
      Specifies whether ``A``'s data is stored in its upper or lower
      triangle. See :ref:`onemkl_datatypes` for more details.

   trans
      Specifies op(``A``), the transposition operation applied to
      ``A``. See :ref:`onemkl_datatypes` for more details. Supported operations are ``transpose::nontrans``
      and ``transpose::conjtrans``.

   n
      The number of rows and columns in ``C``.The value of ``n`` must
      be at least zero.

   k
      Number of columns in op(``A``).

      The value of ``k`` must be at least zero.

   alpha
      Real scaling factor for the rank-k update.

   a
      Pointer to input matrix ``A``.

      .. list-table::
         :header-rows: 1

         * -
           - ``trans`` = ``transpose::nontrans``
           - ``trans`` = ``transpose::trans`` or ``transpose::conjtrans``
         * - Column major
           - ``A`` is an ``n``-by-``k`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``k``.
           - ``A`` is an ``k``-by-``n`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``n``
         * - Row major
           - ``A`` is an ``n``-by-``k`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``n``.
           - ``A`` is an ``k``-by-``n`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``k``.
      
      See :ref:`matrix-storage` for more details.

   lda
      The leading dimension of ``A``. It must be positive.

      .. list-table::
         :header-rows: 1

         * -
           - ``trans`` = ``transpose::nontrans``
           - ``trans`` = ``transpose::trans`` or ``transpose::conjtrans``
         * - Column major
           - ``lda`` must be at least ``n``.
           - ``lda`` must be at least ``k``.
         * - Row major
           - ``lda`` must be at least ``k``.
           - ``lda`` must be at least ``n``.

   beta
      Real scaling factor for matrix ``C``.

   c
      Pointer to input/output matrix ``C``. Must have size at least
      ``ldc``\ \*\ ``n``. See :ref:`matrix-storage` for
      more details.

   ldc
      Leading dimension of ``C``. Must be positive and at least
      ``n``.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   c
      Pointer to the output matrix, overwritten by
      ``alpha``\ \*op(``A``)*op(``A``)\ :sup:`T` +
      ``beta``\ \*\ ``C``. The imaginary parts of the diagonal
      elements are set to zero.

.. container:: section

   .. rubric:: Return Values

   Output event to wait on to ensure computation is complete.

      

   **Parent topic:** :ref:`blas-level-3-routines`
