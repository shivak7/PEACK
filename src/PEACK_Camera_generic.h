#pragma once
#include <PEACK_Device.h>


class PEACKGenCam : public PEACKDevice
{
public:
	int init(int Width, int Height, int FrameRate);
	int init(std::string);
	int getFrames();
	int stop();
	int showFrames();
	int projectPoints(std::vector<float>& From, std::vector<float>& To);	//Function for camera/device specific capability to project points to/from pixels to real world.
	std::string get_device_name();
protected:
	cv::VideoCapture Source;
	int FrameCount;
	std::chrono::time_point<std::chrono::system_clock> startTime;
};
