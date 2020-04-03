.. _syr:

syr
===


.. container::


   Computes a rank-1 update of a symmetric matrix.


   .. container:: section
      :name: GUID-E620D36F-6B4E-40A6-8BDA-3D625DEF55A8


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void syr(queue &exec_queue, uplo upper_lower,      std::int64_t n, T alpha, buffer<T,1> &x, std::int64_t incx,      buffer<T,1> &a, std::int64_t lda)

      ``syr`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``float`` 
         * -  ``double`` 




.. container:: section
   :name: GUID-E154DE4B-4559-4471-B92B-46AF8777AC97


   .. rubric:: Description
      :name: description
      :class: sectiontitle


   The syr routines compute a scalar-vector-vector product add them and
   add the result to a matrix, with a symmetric matrix. The operation is
   defined as


  


      A  <- alpha*x*x :sup:`T` + A


   where:


   ``alpha`` is scalar,


   ``A`` is an ``n``-by-``n`` symmetric matrix,


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


   n
      Number of columns of ``A``. Must be at least zero.


   alpha
      Scaling factor for the matrix-vector product.


   x
      Buffer holding input vector ``x``. The buffer must be of size at
      least (1 + (``n`` - 1)*abs(``incx``)). See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   incx
      Stride of vector ``x``.


   a
      Buffer holding input matrix ``A``. Must have size at least
      ``lda``\ \*\ ``n``. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   lda
      Leading dimension of matrix ``A``. Must be at least ``n``, and
      positive.


.. container:: section
   :name: GUID-C03D1215-FD77-4AD8-8FA2-C48A5D8B938C


   .. rubric:: Output Parameters
      :name: output-parameters
      :class: sectiontitle


   a
      Buffer holding the updated upper triangularpart of the symmetric
      matrix ``A`` if ``upper_lower =upper`` or the updated lower
      triangular part of thesymmetric matrix ``A`` if
      ``upper_lower =lower``.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-2-routines`
      


.. container::

