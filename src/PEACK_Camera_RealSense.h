#pragma once
#include "PEACK_Device.h"
#include <librealsense2/rs.hpp>
#include <thread>
#include <atomic>
#include <opencv2/dnn_superres.hpp>
#include <opencv2/dnn.hpp>


class PEACKrealsense : public PEACKDevice
{
public:
	int init(int Width, int Height, int FrameRate);
	int init(std::string);
	int getFrames();
	int stop();
	int showFrames();
	int projectPoints(std::vector<float> &From, std::vector<float> &To);	//Function for camera/device specific capability to project points to/from pixels to real world.
	
protected:
	rs2::pipeline pipe;
	rs2::device device;
	rs2::playback * playback;
	rs2::config cfg;
	rs2::frame rcolor_frame;
	rs2::frame rdepth_frame;
	rs2_intrinsics intrinsics;
	rs2::decimation_filter dec_filter;
	rs2::spatial_filter spat_filter;
	rs2::temporal_filter temp_filter;
	rs2::disparity_transform depth2disparity;
	
	rs2::frame_queue postprocessed_frames;
	std::atomic_bool alive;

	cv::dnn_superres::DnnSuperResImpl sr;

	rs2::align * align;
	int FrameCount;
};
