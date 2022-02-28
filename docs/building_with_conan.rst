.. _building_with_conan:

Building with Conan
===================

** This method currently works on Linux* only **

** Make sure you have completed :ref:`Build Setup <build_setup>`. **

.. note::
  To understand how dependencies are resolved, refer to "Product and Version
  Information" under
  `Support and Requirements <https://github.com/oneapi-src/oneMKL#support-and-requirements>`_.
  For details about Conan package manager, refer to the
  `Conan Documentation <https://docs.conan.io/en/latest/>`_.

Getting Conan
^^^^^^^^^^^^^

Conan can be `installed <https://docs.conan.io/en/latest/installation.html>`_ from pip:

.. code-block:: bash

   pip3 install conan

Setting up Conan
^^^^^^^^^^^^^^^^

Conan Default Directory
~~~~~~~~~~~~~~~~~~~~~~~

Conan stores all files and data in ``~/.conan``. If you are fine with this
behavior, you can skip to the :ref:`Conan Profiles <conan-profiles>` section.

To change this behavior, set the environment variable ``CONAN_USER_HOME`` to a
path of your choice. A ``.conan/`` directory will be created in this path and
future Conan commands will use this directory to find configuration files and
download dependent packages. Packages will be downloaded into
``$CONAN_USER_HOME/data``. To change the ``"/data"`` part of this directory,
refer to the ``[storage]`` section of ``conan.conf`` file.

To make this setting persistent across terminal sessions, you can add the
line below to your ``~/.bashrc`` or custom runscript. Refer to the
`Conan Documentation <https://docs.conan.io/en/latest/reference/env_vars.html#conan-user-home>`_
for more details.

.. code-block:: sh

   export CONAN_USER_HOME=/usr/local/my_workspace/conan_cache

.. _conan-profiles:

Conan Profiles
~~~~~~~~~~~~~~

Profiles are a way for Conan to determine a basic environment to use for
building a project. This project ships with profiles for:


* Intel(R) oneAPI DPC++ Compiler for x86 CPU and Intel GPU backend: ``inteldpcpp_lnx``


#. Open the profile you wish to use from ``<path to onemkl>/conan/profiles/``
   and set ``COMPILER_PREFIX`` to the path to the root folder of compiler.
   The root folder is the one that contains the ``bin`` and ``lib``
   directories. For example, Intel(R) oneAPI DPC++ Compiler root folder for
   default installation on Linux is
   ``/opt/intel/inteloneapi/compiler/<version>/linux``. The user can define a
   custom path for installing the compiler.

.. code-block:: ini

   COMPILER_PREFIX=<path to Intel(R) oneAPI DPC++ Compiler>


#. 
   You can customize the ``[env]`` section of the profile based on individual
   requirements.

#. 
   Install configurations for this project:

   .. code-block:: sh

      # Inside <path to onemkl>
      $ conan config install conan/

   This command installs all contents of ``<path to onemkl>/conan/``\ , most
   importantly profiles, to conan default directory.

.. note::
  If you change the profile, you must re-run the above command before you can
  use the new profile.

Building
^^^^^^^^

#. 
   Out-of-source build

   .. code-block:: bash

      # Inside <path to onemkl>
      mkdir build && cd build

#. 
   If you choose to build backends with the Intel(R) oneAPI
   Math Kernel Library, install the GPG key as mentioned here:
   https://software.intel.com/en-us/articles/oneapi-repo-instructions#aptpkg

#. 
   Install dependencies

   .. code-block:: sh

      conan install .. --profile <profile_name> --build missing [-o <option1>=<value1>] [-o <option2>=<value2>]

   The ``conan install`` command downloads and installs all requirements for
   the oneMKL DPC++ Interfaces project as defined in
   ``<path to onemkl>/conanfile.py`` based on the options passed. It also
   creates ``conanbuildinfo.cmake`` file that contains information about all
   dependencies and their directories. This file is used in top-level
   ``CMakeLists.txt``.

``-pr | --profile <profile_name>``
Defines a profile for Conan to use for building the project.

``-b | --build <package_name|missing>``
Tells Conan to build or re-build a specific package. If ``missing`` is passed
as a value, all missing packages are built. This option is recommended when
you build the project for the first time, because it caches required packages.
You can skip this option for later use of this command.


#. Build Project
   .. code-block:: sh

      conan build .. [--configure] [--build] [--test]  # Default is all

The ``conan build`` command executes the ``build()`` procedure from
``<path to onemkl>/conanfile.py``. Since this project uses ``CMake``\ , you
can choose to ``configure``\ , ``build``\ , ``test`` individually or perform
all steps by passing no optional arguments.


#. Optionally, you can also install the package. Similar to ``cmake --install . --prefix <install_dir>``.

.. code-block:: sh

   conan package .. --build-folder . --install-folder <install_dir>

``-bf | --build-folder``
Tells Conan where to find the built project.

``-if | --install-folder``
Tells Conan where to install the package. It is similar to specifying ``CMAKE_INSTALL_PREFIX``

.. note::
   For a detailed list of commands and options, refer to the
   `Conan Command Reference <https://docs.conan.io/en/latest/reference/commands.html>`_.

Conan Build Options
^^^^^^^^^^^^^^^^^^^

Backend-Related Options
~~~~~~~~~~~~~~~~~~~~~~~

The following ``options`` are available to pass on ``conan install`` when
building the oneMKL library:


* ``build_shared_libs=[True | False]``. Setting it to ``True`` enables the building of dynamic libraries. The default value is ``True``.
* ``target_domains=[<list of values>]``. Setting it to ``blas`` or any other list of domain(s), enables building of those specific domain(s) only. If not defined, the default value is all supported domains.
* ``enable_mklcpu_backend=[True | False]``. Setting it to ``True`` enables the building of oneMKL mklcpu backend. The default value is ``True``.
* ``enable_mklgpu_backend=[True | False]``. Setting it to ``True`` enables the building of oneMKL mklgpu backend. The default value is ``True``.
* ``enable_mklcpu_thread_tbb=[True | False]``. Setting it to ``True`` enables oneMKL on CPU with TBB threading instead of sequential. The default value is ``True``.

Testing-Related Options
~~~~~~~~~~~~~~~~~~~~~~~

* ``build_functional_tests=[True | False]``. Setting it to ``True`` enables
  the building of functional tests. The default value is ``True``.

Documentation
~~~~~~~~~~~~~

* ``build_doc=[True | False]``. Setting it to ``True`` enables the building of rst files to generate HTML files for updated documentation. The default value is ``False``.

.. note::
  For a mapping between Conan and CMake options, refer to
  :ref:`Building with CMake <building_with_cmake>`.

Example
^^^^^^^

Build oneMKL as a static library for oneMKL cpu and gpu backend:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: sh

   # Inside <path to onemkl>
   mkdir build && cd build
   conan install .. --build missing --profile inteldpcpp_lnx -o build_shared_libs=False
   conan build ..
