#pragma once
#include <PEACK_tracker.h>
#include <PEACK_math.h>
#include <median_filt.h>

#define NUM_UPPERLIMBS 6
#define TRACK_FILTER_ORD 5

void MyEllipse(cv::Mat& img, int cx, int cy, int w, int h, double angle);

struct ULStruct
{
	std::vector<float> LShldr, RShldr, LElb, RElb, LWrist, RWrist;
};

struct BBox
{
	cv::RotatedRect LForeArm;
	cv::RotatedRect RForeArm;
	cv::RotatedRect LUpperArm;
	cv::RotatedRect RUpperArm;
};

class ULTracker
{
	PEACKTracker * T;
	bool UseFilter;
	ULStruct Pre;
	ULStruct Current;
	ULStruct Target;
	//std::vector<float> PreLShldr, PreRShldr, PreLElb, PreRElb, PreLWrist, PreRWrist;
	MedianFilter <float> Filters[NUM_UPPERLIMBS * 3];
	int calculateBBoxes(ULStruct&,BBox&);
public:
	BBox ActualBBox;
	BBox TargetBBox;
	ULTracker(PEACKTracker* tracker, bool filtON);
	int Init(PEACKTracker* tracker, bool filtON);
	int Update();
	int UpdateFilters();
	cv::Mat DrawBBoxes(cv::Mat&, BBox&);
	int	DrawJointsHands(cv::Mat&, ULStruct&);
	int GeneratePose(int);
	
};