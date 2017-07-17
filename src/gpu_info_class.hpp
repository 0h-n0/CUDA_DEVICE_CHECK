#ifndef _GPU_INFO_CLASS_HPP
#define _GPU_INFO_CLASS_HPP

#include <string>

class GPU
{
public:
    GPU(int device_id);
    void getDeviceInfo(int);
    void showDeviceInfo(void);
private:
    int device_id;
    int driverVersion;
    int runtimeVersion;
    std::string name;

    
};
#endif
