#pragma once
#include "PEACK_tracker.h"
#define OPENPOSE_FLAGS_DISABLE_PRODUCER
//#define OPENPOSE_FLAGS_DISABLE_DISPLAY
// OpenPose dependencies
#include <openpose/headers.hpp>
#include <chrono>

class PEACKTracker_OpenPose : public PEACKTracker
{
public:
	int init(int target, int mode, PEACKDevice *, std::string FileName);
	int getFrameFromDevice();
	int predictKeypoints();
	int processKeypoints2D();
	int processKeypoints3D();
	int getPartKeyPoint(int,std::vector<float>&);
	~PEACKTracker_OpenPose();

private:
	std::chrono::time_point<std::chrono::high_resolution_clock> opTimer;
	op::Wrapper opWrapper{ op::ThreadManagerMode::Asynchronous };
	std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>> datumProcessed;
	cv::VideoWriter video;
	
};
