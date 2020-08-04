#pragma once
#include "PEACK_tracker.h"
#define OPENPOSE_FLAGS_DISABLE_PRODUCER
//#define OPENPOSE_FLAGS_DISABLE_DISPLAY
// OpenPose dependencies
#include <cubemos/engine.h>
#include <cubemos/skeleton_tracking.h>
#include <chrono>


using CUBEMOS_SKEL_Buffer_Ptr = std::unique_ptr<CM_SKEL_Buffer, void (*)(CM_SKEL_Buffer*)>;

class PEACKTracker_CubeMos : public PEACKTracker
{
public:
	//PEACKTracker_CubeMos(int target, int mode, PEACKDevice*, std::string FileName);
	int init(int target, int mode, PEACKDevice*, std::string FileName);
	int getFrameFromDevice();
	int predictKeypoints();
	int processKeypoints2D();
	int processKeypoints3D();
	int getPartKeyPoint(int, std::vector<float>&);
	~PEACKTracker_CubeMos();
	int release();
	PEACKTracker_CubeMos() :skeletonsPresent(CUBEMOS_SKEL_Buffer_Ptr(new CM_SKEL_Buffer(), [](CM_SKEL_Buffer* pb) {
		cm_skel_release_buffer(pb);
		delete pb;
		})), skeletonsLast(CUBEMOS_SKEL_Buffer_Ptr(new CM_SKEL_Buffer(), [](CM_SKEL_Buffer* pb) {
			cm_skel_release_buffer(pb);
			delete pb;
			})) {}

	int getMappingFromString(std::string);
	std::string getMappingFromIndex(unsigned int);
	int populateMap();
private:
	CM_TargetComputeDevice enInferenceMode;
	CM_SKEL_Handle* handle;
	CM_ReturnCode retCode;
	cv::VideoWriter video;
	const int nHeight = 192;	// Originally 192
	CM_SKEL_AsyncRequestHandle* skeletRequestHandle;
	CM_Image imageLast;
	CUBEMOS_SKEL_Buffer_Ptr skeletonsPresent;
	CUBEMOS_SKEL_Buffer_Ptr skeletonsLast;
	int nTimeoutMs;
};