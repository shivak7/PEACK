#pragma once
#include "PEACK_Device.h"
#include <librealsense2/rs.hpp>

class PEACKrealsense_Rosbag : public PEACKDevice
{
public:
	int init(std::string Filename);
	int getFrames();
	int stop();
	int showFrames();
	int projectPoints(std::vector<float> &From, std::vector<float> &To);	//Function for camera/device specific capability to project points to/from pixels to real world.
	std::string FileName;
protected:
	rs2::pipeline pipe;
	rs2::config cfg;
	rs2::frame rcolor_frame;
	rs2::frame rdepth_frame;
	rs2_intrinsics intrinsics;
	rs2::align * align;
};


