.. _project_cleanup:

Project Cleanup
===============

Most use-cases involve building the project without the need to cleanup the
build directory. However, if you wish to cleanup the build directory, you can
delete the ``build`` folder and create a new one. If you wish to cleanup the
build files but retain the build configuration, following commands will help
you do so. They apply to both ``Conan`` and ``CMake`` methods of building
this project.

.. code-block:: sh

   # If you use "GNU/Unix Makefiles" for building,
   make clean

   # If you use "Ninja" for building
   ninja -t clean
