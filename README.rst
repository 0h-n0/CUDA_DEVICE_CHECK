CUDA_DEVICE_CHECK
==================

1. stress test on your nvidia device.
2. check memory allocation speed.
3. check data transfer speed between host and device.
4. check calculation speed.
   1. cuDNN.
   2. cuBLAS

REQUIREMENTS
--------------

* cmake > 3.9.0
* boost > 1.65.0
* cuda toolkit > 8.0
* cuDNN > 6.0  


HOW TO GET  
---------------

.. code-block:: shell

   git clone --recursive https://github.com/0h-n0/CUDA_DEVICE_CHECK.git

OR                
^^^^^^

.. code-block:: shell

   git clone https://github.com/0h-n0/CUDA_DEVICE_CHECK.git
   git submodule init
   git submodule update


BUILD
----------

.. code-block:: shell

   cd CUDA_DEVICE_CHECK
   mkdir build
   cd build
   cmake ..
   make -j $(porc)

RUN
---------



REFERENCE
-----------

* http://wili.cc/blog/gpu-burn.html
