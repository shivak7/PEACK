#include "PEACKDevice_RealSense_Rosbag.h"

int PEACKrealsense_Rosbag::init(std::string fn)
{
	FileName = fn;

	std::cout << "Using device: Realsense Rosbag file " << FileName<< std::endl;

	AnyFrames = 0;
	
	align = new rs2::align(RS2_STREAM_DEPTH);
	
	cfg.enable_device_from_file(FileName);
	pipe.start(cfg);

	rs2::frameset frames;
	frames = pipe.wait_for_frames();
	rs2::depth_frame d = frames.get_depth_frame();

	FrameWidth = d.get_width();
	FrameHeight = d.get_height();
	FrameRate = -1;	//File

	return 0;
}
