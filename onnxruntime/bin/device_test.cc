#include <portaudio.h>
#include <iostream>

int main()
{
    Pa_Initialize();

    int devices = Pa_GetDeviceCount();
    std::cout << devices << std::endl;

    for (int i = 0; i != devices; ++i)
    {
        auto * info = Pa_GetDeviceInfo(i);
        std::cout << info->name << std::endl;
    }

    Pa_Terminate();
}