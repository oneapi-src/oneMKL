.. _onemkl_blas_gemm_bias:

gemm_bias
=========

Computes a matrix-matrix product using general integer matrices with bias.

.. _onemkl_blas_gemm_bias_description:

.. rubric:: Description

The gemm_bias routines compute a scalar-matrix-matrix product and
add the result to a scalar-matrix product, using general integer matrices with biases/offsets. 
The operation is defined as:

.. math::
      
      \scriptstyle C \leftarrow alpha*(op(A) - A\_offset)*(op(B) - B\_offset) + beta*C + C\_offset

where:

op(``X``) is one of op(``X``) = ``X``, or op(``X``) = ``X``\ :sup:`T`, or
op(``X``) = ``X``\ :sup:`H`,

``alpha`` and ``beta`` are scalars,

``A_offset`` is an ``m``-by-``k`` matrix with every element equal to the value ao,

``B_offset`` is a ``k``-by-``n`` matrix with every element equal to the value bo,

``C_offset`` is an ``m``-by-``n`` matrix defined by the 
co buffer as described below, 

``A``, ``B``, and ``C`` are matrices,

op(``A``) is ``m`` x ``k``, op(``B``) is ``k`` x ``n``, and
``C`` is ``m`` x ``n``.

``gemm_bias`` supports the following precisions.

  .. list-table:: 
     :header-rows: 1

     * -  Ts 
       -  Ta 
       -  Tb 
       -  Tc 
     * -  ``float`` 
       -  ``std::uint8_t`` 
       -  ``std::uint8_t`` 
       -  ``std::int32_t`` 
     * -  ``float`` 
       -  ``std::int8_t`` 
       -  ``std::uint8_t`` 
       -  ``std::int32_t`` 
     * -  ``float`` 
       -  ``std::uint8_t`` 
       -  ``std::int8_t`` 
       -  ``std::int32_t`` 
     * -  ``float`` 
       -  ``std::int8_t`` 
       -  ``std::int8_t`` 
       -  ``std::int32_t`` 

.. _onemkl_blas_gemm_bias_buffer:

gemm_bias (Buffer Version)
--------------------------

.. rubric:: Syntax
      
.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       void gemm_bias(sycl::queue &queue,
                      onemkl::transpose transa,
                      onemkl::transpose transb,
                      onemkl::offset offset_type,
                      std::int64_t m,
                      std::int64_t n,
                      std::int64_t k,
                      Ts alpha,
                      sycl::buffer<Ta,1> &a,
                      std::int64_t lda,
                      Ta ao,
                      sycl::buffer<Tb,1> &b,
                      std::int64_t ldb,
                      Tb bo,
                      Ts beta,
                      sycl::buffer<Tc,1> &c,
                      std::int64_t ldc,
                      sycl::buffer<Tc,1> &co)
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       void gemm_bias(sycl::queue &queue,
                      onemkl::transpose transa,
                      onemkl::transpose transb,
                      onemkl::offset offset_type,
                      std::int64_t m,
                      std::int64_t n,
                      std::int64_t k,
                      Ts alpha,
                      sycl::buffer<Ta,1> &a,
                      std::int64_t lda,
                      Ta ao,
                      sycl::buffer<Tb,1> &b,
                      std::int64_t ldb,
                      Tb bo,
                      Ts beta,
                      sycl::buffer<Tc,1> &c,
                      std::int64_t ldc,
                      sycl::buffer<Tc,1> &co)
   }
      
.. container:: section
   
   .. rubric:: Input Parameters
 
   queue
      The queue where the routine should be executed.
 
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
 
   offset_type
      Specifies the form of ``C_offset`` used in the matrix
      multiplication. See
      :ref:`onemkl_datatypes` for
      more details.
 
   m
      Number of rows of op(``A``) and ``C``. Must be at least zero.
 
   n
      Number of columns of op(``B``) and ``C``. Must be at least
      zero.
 
   k
      Number of columns of op(``A``) and rows of op(``B``). Must be
      at least zero.
 
   alpha
      Scaling factor for the matrix-matrix product.
 
   a
      The buffer holding the input matrix ``A``.

      .. list-table::
         :header-rows: 1

         * -
           - ``A`` not transposed
           - ``A`` transposed
         * - Column major
           - ``A`` is an ``m``-by-``k`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``k``.
           - ``A`` is an ``k``-by-``m`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``m``
         * - Row major
           - ``A`` is an ``m``-by-``k`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``m``.
           - ``A`` is an ``k``-by-``m`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``k``

      See :ref:`matrix-storage` for more details.
 
   lda
      The leading dimension of ``A``. It must be positive.

      .. list-table::
         :header-rows: 1

         * -
           - ``A`` not transposed
           - ``A`` transposed
         * - Column major
           - ``lda`` must be at least ``m``.
           - ``lda`` must be at least ``k``.
         * - Row major
           - ``lda`` must be at least ``k``.
           - ``lda`` must be at least ``m``.
 
   ao 
      Specifies the scalar offset value for matrix ``A``.
 
   b
      Buffer holding the input matrix ``B``.

      .. list-table::
         :header-rows: 1

         * -
           - ``B`` not transposed
           - ``B`` transposed
         * - Column major
           - ``B`` is an ``k``-by-``n`` matrix so the array ``b``
             must have size at least ``ldb``\ \*\ ``n``.
           - ``B`` is an ``n``-by-``k`` matrix so the array ``b``
             must have size at least ``ldb``\ \*\ ``k``
         * - Row major
           - ``B`` is an ``k``-by-``n`` matrix so the array ``b``
             must have size at least ``ldb``\ \*\ ``k``.
           - ``B`` is an ``n``-by-``k`` matrix so the array ``b``
             must have size at least ``ldb``\ \*\ ``n``
   
      See :ref:`matrix-storage` for more details.
 
   ldb
      The leading dimension of ``B``. It must be positive.

      .. list-table::
         :header-rows: 1

         * -
           - ``B`` not transposed
           - ``B`` transposed
         * - Column major
           - ``ldb`` must be at least ``k``.
           - ``ldb`` must be at least ``n``.
         * - Row major
           - ``ldb`` must be at least ``n``.
           - ``ldb`` must be at least ``k``.
 
   bo 
      Specifies the scalar offset value for matrix ``B``.
 
   beta
      Scaling factor for matrix ``C``.
 
   c
      Buffer holding the input/output matrix ``C``.  It must have a
      size of at least ``ldc``\ \*\ ``n`` if column major layout is
      used to store matrices or at least ``ldc``\ \*\ ``m`` if row
      major layout is used to store matrices . 
      See :ref:`matrix-storage` for more details.
 
   ldc
      The leading dimension of ``C``. It must be positive and at least
      ``m`` if column major layout is used to store matrices or at
      least ``n`` if row major layout is used to store matrices.
      
   co
      Buffer holding the offset values for matrix ``C``.
 
      If ``offset_type`` = ``offset::fix``, the ``co`` array must have
      size at least 1.
 
 
      If ``offset_type`` = ``offset::col``, the ``co`` array must have
      size at least ``max(1,m)``.
 
 
      If ``offset_type`` = ``offset::row``, the ``co`` array must have
      size at least ``max(1,n)``. 
 
.. container:: section
 
   .. rubric:: Output Parameters
 
   c
      Output buffer, overwritten by ``alpha`` * (op(``A``) -
      ``A_offset``)*(op(``B``) - ``B_offset``) + ``beta`` * ``C`` + ``C_offset``.
 
.. container:: section
 
   .. rubric:: Notes
 
   If ``beta`` = 0, matrix ``C`` does not need to be initialized
   before calling ``gemm_bias``.


.. _onemkl_blas_gemm_bias_usm:

gemm_bias (USM Version)
-----------------------

.. rubric:: Syntax
      
.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       sycl::event gemm_bias(sycl::queue &queue,
                             onemkl::transpose transa,
                             onemkl::transpose transb,
                             onemkl::offset offset_type,
                             std::int64_t m,
                             std::int64_t n,
                             std::int64_t k,
                             Ts alpha,
                             const Ta *a,
                             std::int64_t lda,
                             Ta ao,
                             const Tb *b,
                             std::int64_t ldb,
                             Tb bo,
                             Ts beta,
                             Tc *c,
                             std::int64_t ldc,
                             const Tc *co,
                             const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       sycl::event gemm_bias(sycl::queue &queue,
                             onemkl::transpose transa,
                             onemkl::transpose transb,
                             onemkl::offset offset_type,
                             std::int64_t m,
                             std::int64_t n,
                             std::int64_t k,
                             Ts alpha,
                             const Ta *a,
                             std::int64_t lda,
                             Ta ao,
                             const Tb *b,
                             std::int64_t ldb,
                             Tb bo,
                             Ts beta,
                             Tc *c,
                             std::int64_t ldc,
                             const Tc *co,
                             const std::vector<sycl::event> &dependencies = {})
   }
      
.. container:: section
   
   .. rubric:: Input Parameters
 
   queue
      The queue where the routine should be executed.
 
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
 
   offset_type
      Specifies the form of ``C_offset`` used in the matrix
      multiplication. See
      :ref:`onemkl_datatypes` for
      more details.
 
   m
      Number of rows of op(``A``) and ``C``. Must be at least zero.
 
   n
      Number of columns of op(``B``) and ``C``. Must be at least
      zero.
 
   k
      Number of columns of op(``A``) and rows of op(``B``). Must be
      at least zero.
 
   alpha
      Scaling factor for the matrix-matrix product.
 
   a
      Pointer to input matrix ``A``.
 
      .. list-table::
         :header-rows: 1

         * -
           - ``A`` not transposed
           - ``A`` transposed
         * - Column major
           - ``A`` is an ``m``-by-``k`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``k``.
           - ``A`` is an ``k``-by-``m`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``m``
         * - Row major
           - ``A`` is an ``m``-by-``k`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``m``.
           - ``A`` is an ``k``-by-``m`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``k``
 
      See :ref:`matrix-storage` for more details.
 
   lda
      The leading dimension of ``A``. It must be positive.

      .. list-table::
         :header-rows: 1

         * -
           - ``A`` not transposed
           - ``A`` transposed
         * - Column major
           - ``lda`` must be at least ``m``.
           - ``lda`` must be at least ``k``.
         * - Row major
           - ``lda`` must be at least ``k``.
           - ``lda`` must be at least ``m``.
 
   ao
      Specifies the scalar offset value for matrix ``A``.
 
   b
      Pointer to input matrix ``B``.
 
      .. list-table::
         :header-rows: 1

         * -
           - ``B`` not transposed
           - ``B`` transposed
         * - Column major
           - ``B`` is an ``k``-by-``n`` matrix so the array ``b``
             must have size at least ``ldb``\ \*\ ``n``.
           - ``B`` is an ``n``-by-``k`` matrix so the array ``b``
             must have size at least ``ldb``\ \*\ ``k``
         * - Row major
           - ``B`` is an ``k``-by-``n`` matrix so the array ``b``
             must have size at least ``ldb``\ \*\ ``k``.
           - ``B`` is an ``n``-by-``k`` matrix so the array ``b``
             must have size at least ``ldb``\ \*\ ``n``
 
      See :ref:`matrix-storage` for more details.
 
   ldb
      The leading dimension of ``B``. It must be positive.

      .. list-table::
         :header-rows: 1

         * -
           - ``B`` not transposed
           - ``B`` transposed
         * - Column major
           - ``ldb`` must be at least ``k``.
           - ``ldb`` must be at least ``n``.
         * - Row major
           - ``ldb`` must be at least ``n``.
           - ``ldb`` must be at least ``k``.
 
   bo 
      Specifies the scalar offset value for matrix ``B``.
 
   beta
      Scaling factor for matrix ``C``.
 
   c
      Pointer to input/output matrix ``C``. It must have a
      size of at least ``ldc``\ \*\ ``n`` if column major layout is
      used to store matrices or at least ``ldc``\ \*\ ``m`` if row
      major layout is used to store matrices . See :ref:`matrix-storage` for more details.
 
   ldc
      The leading dimension of ``C``. It must be positive and at least
      ``m`` if column major layout is used to store matrices or at
      least ``n`` if row major layout is used to store matrices.

   co
      Pointer to offset values for matrix ``C``.
 
 
      If ``offset_type`` = ``offset::fix``, the ``co`` array must have
      size at least 1.
 
 
      If ``offset_type`` = ``offset::col``, the ``co`` array must have
      size at least ``max(1,m)``.
 
 
      If ``offset_type`` = ``offset::row``, the ``co`` array must have
      size at least ``max(1,n)``.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.
 
.. container:: section
 
   .. rubric:: Output Parameters
 
   c
      Pointer to the output matrix, overwritten by ``alpha`` * (op(``A``) -
      ``A_offset``)*(op(``B``) - ``B_offset``) + ``beta`` * ``C`` + ``C_offset``.
 
.. container:: section
 
   .. rubric:: Notes
 
   If ``beta`` = 0, matrix ``C`` does not need to be initialized
   before calling ``gemm_bias``.

.. container:: section

   .. rubric:: Return Values

   Output event to wait on to ensure computation is complete.


   **Parent topic:** :ref:`blas-like-extensions`
