/*
 * 检测输入的麦克风设备是否正常接入。
 * */

#include <portaudio.h>
#include <iostream>

int main()
{
    Pa_Initialize();

    int devices = Pa_GetDeviceCount();
    if (devices==0){
        std::cout << "Not find audio input device." << std::endl;
    }

    for (int i = 0; i != devices; ++i)
    {
        auto * info = Pa_GetDeviceInfo(i);
        std::cout << info->name << std::endl;
    }

    Pa_Terminate();
}