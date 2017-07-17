#include <iostream>
#include <thread>
#include <memory>
#include <cuda_runtime.h>
#include <cudnn.h>
//#include <boost/program_options.hpp>
#include "spdlog/spdlog.h"

using namespace std;
using namespace boost::program_options;
namespace spd = spdlog;

//////////////////////
// kernel functions //
//////////////////////


///////////
// Class //
///////////

template <typename T>
class GPUTest 
{
public:
    GPUTest(const int id);
    void allocate();
    void graduallyAllocate();
    void deviceInfo();
    virtual ~GPUTest();
    void setLog(void);
    std::shared_ptr<spd::logger> console;
private:
    long long int size;
    int device_id;
    // for device prof;
    
};

template <typename T>
void GPUTest<T>::setLog(void)
{
    try {
        // console->info(function_name, message);
        console = spd::stdout_color_mt("console");
        spd::set_level(spd::level::debug);
        console->info("Set console log.");
    }catch (const spd::spdlog_ex& ex){
        cout  << "Log init failed: " << ex.what() << endl;
    }
}

template <typename T>
GPUTest<T>::GPUTest(const int id)
{
    cudaSetDevice(id);
    device_id = id;
}

template <typename T>
GPUTest<T>::~GPUTest()
{
    cudaDeviceReset();
}

template <typename T>
void GPUTest<T>::allocate(void)
{
    cout << "GPUTEST" << endl;
}

template <typename T>
void GPUTest<T>::graduallyAllocate(void)
{
    console->info("TEST: Gradual increasment of Memory Allocation on gpu device.");
    int mb_size = 1 << 10 << 10;
    int increase_factor = 1 << 5;
    float *fd, *fh;
    int *id, *ih;
    char *cd, *ch;
    int maxsize = 8;
    console->info("TEST: MAX Allocation Size {} GB on gpu device.", maxsize);
    console->info("SET:  Increase factor {} MB on gpu device.", increase_factor);

    console->info("TEST: char[{}bytes].", sizeof(char));
    for(int i = 0 ;i < maxsize; ++i){
        
    }
    console->info("SLEEP: {} seconds.", 20);    
    std::this_thread::sleep_for(20s);

    

    console->info("TEST: int[{}bytes].", sizeof(int));
    for(int i = 0 ;i < maxsize; ++i){
        
    }
    console->info("SLEEP: {} seconds.", 20);    
    std::this_thread::sleep_for(20s);
    
    console->info("TEST: float[{}bytes].", sizeof(float));
    for(int i = 0 ;i < maxsize; ++i){
        
    }
    console->info("SLEEP: {} seconds.", 20);    
    std::this_thread::sleep_for(20s);


    console->info("TEST: dobule[{}bytes].", sizeof(double));
    for(int i = 0 ;i < maxsize; ++i){
        
    }

    
    
    

    cudaDeviceSynchronize();
    
}

template <typename T>
void GPUTest<T>::deviceInfo(void)
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
    int deviceId=0;

    // To change argments parser from boost to Taywee/args?
    /* 
   options_description options1("This programm does GPU stress test.");
    options1.add_options()
        ("help,h",    "help mesage.")
        ("deviceid,d", value<int>(),   "set DeviceId of GPU.");
        //("memory_allocation_size,s",  "set Memory allocation size (Mb).");
    
    variables_map values;
    try{
        store(parse_command_line(argc, argv, options1), values);
        notify(values);
        if (values.count("help")) {
			cout << options1 << endl;
            exit(EXIT_FAILURE);
        }
        if (!values.count("deviceid")) {
			// options_description は標準出力に投げることが出来る
			cout << options1 << endl;
            exit(EXIT_FAILURE);
		}
		if (values.count("deviceid"))
            cout << "set DeviceId: " <<  endl;
            //cout << "set DeviceId: " << values["deviceid"].as<string>() << endl;
            
    }catch(std::exception &e){
        std::cout << e.what() << std::endl;
        exit(EXIT_FAILURE);        
    }
    */


    GPUTest<int> g(deviceId);
    g.setLog();
    //g.deviceInfo();
    g.graduallyAllocate();

    return 0;
}

