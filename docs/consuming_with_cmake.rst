.. _consuming_with_cmake:

Consuming with CMake
==================== 

This can be done in two ways:

* By using ``find_package`` with an installed version of oneMKL interface library.
* By downloading and building oneMKL as part of your application using FetchContent_.


.. _using_from_installed_binary:

Using an installed oneMKL Interface Library
###########################################

If the oneMKL interfaces have been previously installed, either by building from source or as a distributed
binary, they can be consumed using CMake using ``find_package(oneMKL REQUIRED)``. The compiler used
for the target library or application should match that used to build oneMKL interface library.

For example:

.. code-block:: cmake

    find_package(oneMKL REQUIRED)
    target_link_libraries(myTarget PRIVATE MKL::onemkl)

Different targets can be used depending on the requirements of oneMKL. 
To link against the entire library, the ``MKL::onemkl`` target should be used.
For specific domains, ``MKL::onemkl_<domain>`` should be used.
And for specific backends, ``MKL::onemkl_<domain>_<backend>`` should be used.

When using a binary, it may be useful to know the backends that were enabled during the build.
To check for the existence of backends, CMake's ``if(TARGET <target>)`` construct can be used.
For example, with the ``cufft`` backend:

.. code-block:: cmake

    if(TARGET MKL::onemkl_dft_cufft)
        target_link_libraries(myTarget PRIVATE MKL::onemkl_dft_cufft)
    else()
        message(FATAL_ERROR "oneMKL interface was not built with CuFFT backend")
    endif()


If oneMKL has been installed to a non-standard location, the operating system
may not find the backend libraries when they're lazily loaded at runtime. 
To make sure they're found you may need to set ``LD_LIBRARY_PATH=<onemkl_install_dir>/lib:$LD_LIBRARY_PATH``
on Linux.

.. _using_with_fetchcontent:

Using CMake's FetchContent
##########################


The FetchContent_ functionality of CMake can be used to download, build and install oneMKL 
interface library as part of the build.

For example:

.. code-block:: cmake

    include(FetchContent)
    set(BUILD_FUNCTIONAL_TESTS OFF)
    set(BUILD_EXAMPLES OFF)
    set(ENABLE_<BACKEND_NAME>_BACKEND ON)
    FetchContent_Declare(
            onemkl_interface_library
            GIT_REPOSITORY https://github.com/oneapi-src/oneMKL.git
            GIT_TAG develop
    )
    FetchContent_MakeAvailable(onemkl_interface_library)

    target_link_libraries(myTarget PRIVATE onemkl)

The build parameters should be appropriately set before ``FetchContent_Declare``.
See :ref:`building_the_project_with_dpcpp` or :ref:`building_the_project_with_adaptivecpp`.

To link against the main library with run-time dispatching, use the target ``onemkl``.
To link against particular domains, use the target ``onemkl_<domain>``. For example, ``onemkl_blas`` or ``onemkl_dft``.
To link against particular backends (as required for static dispatch of oneAPI calls to a particular backend),
use the target ``onemkl_<domain>_<backend>``. For example, ``onemkl_dft_cufft``.

When using the run-time dispatch mechanism, it is likely that the operating system
will not find the backend libraries when they're loaded at runtime. 
To make sure they're found you may need to set ``LD_LIBRARY_PATH=<onemkl_install_dir>/lib:$LD_LIBRARY_PATH``
on Linux.


.. _FetchContent: https://cmake.org/cmake/help/latest/module/FetchContent.html
