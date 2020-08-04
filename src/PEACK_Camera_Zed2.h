#pragma once
#include <PEACK_Device.h>
#include <sl/Camera.hpp>

class PEACKzed2 : public PEACKDevice
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
	sl::Camera zed2;
	sl::InitParameters init_parameters;
	sl::Mat zcolor_frame;
	sl::Mat zdepth_frame;
	sl::Mat zpoint_cloud;


	int FrameCount;
};

