/////////////////////////////////////////////////////////////////////////////
// This application check how mach memory can be allocated on your device. //
// The amout of allocated memory gradually is increased in this program.   //
// Once allocateion reach to max size, Iterarions is stopped and your can  //
// stress on your device via another application                           //
/////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <memory>
#include <cuda_runtime.h>
#include <cudnn.h>
//#include <boost/program_options.hpp>
#include "spdlog/spdlog.h"

using namespace std;
//using namespace boost::program_options;
namespace spd = spdlog;

//////////////////////
// kernel functions //
//////////////////////

///////////
// Class //
///////////
static int number = 0;
template <typename T>
class GPUTest 
{
public:
    GPUTest(const int id);
    void graduallyAllocate();
    void showDeviceInfo();
    void getDeviceInfo();
    virtual ~GPUTest();
    void _setLog(void);
    std::shared_ptr<spd::logger> console;
    cudaError_t err;
private:
    long long int size;
    int deviceID;
    unsigned long long int totalGlobalMem;
    size_t freeMem, totalMem;
    
    // for device prof;
};

template <typename T>
void GPUTest<T>::_setLog(void)
{
    try {
        number++;
        // console->info(function_name, message);
        string s = "GPUTest" + to_string(number);
        console = spd::stdout_color_mt(s);
        spd::set_level(spd::level::debug);
        console->info("SET: console log.");

    }catch (const spd::spdlog_ex& ex){
        cout  << "Log init failed: " << ex.what() << endl;
    }
}

template <typename T>
GPUTest<T>::GPUTest(const int id)
{
    this->_setLog();
    console->info("SET: GPU Device ID [{}].", id);    
    cudaSetDevice(id);
    deviceID = id;
}

template <typename T>
GPUTest<T>::~GPUTest()
{
    cudaDeviceReset();
}

template <typename T>
void GPUTest<T>::graduallyAllocate(void)
{
    console->info("TEST: Gradual increasment of Memory Allocation on gpu device.");
    size_t mb_size = 1 << 10 << 10;
    int increase_factor = 1 << 3;
    T *fd, *fh;
    int maxsize = 8;
    int i = 1;
    int total = 0;
    int sleep_seconds = 60;
    console->info("TEST: MAX Allocation Size {} GB on gpu device.", maxsize);
    console->info("SET:  Increase factor {} MB on gpu device.", increase_factor);
    console->info("TEST: unit = [{}bytes].", sizeof(T));
    console->info("Increasement scedule is [mb_unit [{}*1024*1024] * Increase factor[{}] * time step]."
                  , sizeof(T), increase_factor);
    while(maxsize * 1024 > sizeof(T) * increase_factor * i){
        console->info("Allocation: {} MB on GPU. ", sizeof(T) * increase_factor * i);
        total +=  sizeof(T) * increase_factor * 1024 * 1024 * i;
        err = cudaMalloc((void **)&fd, sizeof(T) * increase_factor * 1024 * 1024 * i);
        if(cudaSuccess != err){
            console->info("Error: can't allocate {} MB on GPU.", sizeof(T) * increase_factor * i);            
            err = cudaMalloc((void **)&fd, sizeof(T) * increase_factor * 1024 * 1024 * --i);            
            console->info("Reallocation: {} MB on GPU. ", sizeof(T) * increase_factor * i);
            console->info("SLEEP: {} seconds.", sleep_seconds);
            console->info("CHECK: stress on your device via another application.");                
            std::this_thread::sleep_for(std::chrono::seconds(sleep_seconds));
            cudaFree(fd);        
            break;
        }
        cudaFree(fd);                
        ++i;
    }
    cudaDeviceSynchronize();
}


template <typename T>
void GPUTest<T>::getDeviceInfo(void)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceID);
    totalGlobalMem = deviceProp.totalGlobalMem;
    console->info("Total amount of global memory:{} GBytes ({} bytes)",
                  (float)totalGlobalMem / pow(1024.0, 3), (unsigned long long)totalGlobalMem);

    cudaMemGetInfo(&freeMem, &totalMem);
    console->info("Free Mem:{}MB, Total Mem:{}MB", int(freeMem/1024/1024), int(totalMem/1024/1024));
    
    
}

template <typename T>
void GPUTest<T>::showDeviceInfo(void)
{
    int dev = 0, driverVersion = 0, runtimeVersion = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Device %d: \"%s\"\n", dev, deviceProp.name);
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
           driverVersion / 1000, (driverVersion % 100) / 10,
           runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
           deviceProp.major, deviceProp.minor);
    printf("  Total amount of global memory:                 %.2f GBytes (%llu "
           "bytes)\n", (float)deviceProp.totalGlobalMem / pow(1024.0, 3),
           (unsigned long long)deviceProp.totalGlobalMem);
    printf("  GPU Clock rate:                                %.0f MHz (%0.2f "
           "GHz)\n", deviceProp.clockRate * 1e-3f,
           deviceProp.clockRate * 1e-6f);
    printf("  Memory Clock rate:                             %.0f Mhz\n",
           deviceProp.memoryClockRate * 1e-3f);
    printf("  Memory Bus Width:                              %d-bit\n",
           deviceProp.memoryBusWidth);

    if (deviceProp.l2CacheSize)
    {
        printf("  L2 Cache Size:                                 %d bytes\n",
               deviceProp.l2CacheSize);
    }
    printf("  Max Texture Dimension Size (x,y,z)             1D=(%d), "
           "2D=(%d,%d), 3D=(%d,%d,%d)\n", deviceProp.maxTexture1D,
           deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
           deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1],
           deviceProp.maxTexture3D[2]);
    printf("  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, "
           "2D=(%d,%d) x %d\n", deviceProp.maxTexture1DLayered[0],
           deviceProp.maxTexture1DLayered[1], deviceProp.maxTexture2DLayered[0],
           deviceProp.maxTexture2DLayered[1],
           deviceProp.maxTexture2DLayered[2]);
    printf("  Total amount of constant memory:               %lu bytes\n",
           deviceProp.totalConstMem);
    printf("  Total amount of shared memory per block:       %lu bytes\n",
           deviceProp.sharedMemPerBlock);
    printf("  Total number of registers available per block: %d\n",
           deviceProp.regsPerBlock);
    printf("  Warp size:                                     %d\n",
           deviceProp.warpSize);
    printf("  Maximum number of threads per multiprocessor:  %d\n",
           deviceProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of threads per block:           %d\n",
           deviceProp.maxThreadsPerBlock);
    printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
           deviceProp.maxThreadsDim[0],
           deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);
    printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
           deviceProp.maxGridSize[0],
           deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);
    printf("  Maximum memory pitch:                          %lu bytes\n",
           deviceProp.memPitch);
}


///////////////////
// parse options //
///////////////////




///////////////////
// main function //
///////////////////

int main(int argc, char *argv[])
{
    int deviceID=1;



    // TODO: make argments parser.
    GPUTest<int> *g = new GPUTest<int>(deviceID);
    g->getDeviceInfo();
    //g->graduallyAllocate();

    GPUTest<float> *fg = new GPUTest<float>(deviceID);
    fg->getDeviceInfo();
    fg->graduallyAllocate();

    GPUTest<double> *dg = new GPUTest<double>(deviceID);
    dg->getDeviceInfo();
    dg->graduallyAllocate();

    GPUTest<char> *cg = new GPUTest<char>(deviceID);
    cg->getDeviceInfo();
    cg->graduallyAllocate();
    
    cudaDeviceReset();
    
    return 0;
}

