#pragma once
#include <string>
#include <opencv2\opencv.hpp>
#include <vector>

#define PEACK_DEVICE_FILE 1
#define PEACK_DEVICE_CAMERA 2

#define PEACK_DEVICE_STATUS_RUNNING 1
#define PEACK_DEVICE_STATUS_STOPPED 2

#define PEACK_DEVICE_DIM_2D 1
#define PEACK_DEVICE_DIM_3D 2

class PEACKDevice {
public:
	std::string CamModel;
	std::string InputFile;
	cv::Mat ColorFrame;
	cv::Mat DepthFrame;
	int FrameWidth;
	unsigned int Dim;
	int FrameHeight;
	int FrameRate;
	bool AnyFrames;
	int Device_Type;
	int CurrentStatus;
	double TimeStamp;
	double FirstTimeStamp;
	virtual int init(int Width, int Height, int FrameRate)=0;
	virtual int init(std::string) = 0;	//for using files as devices
	virtual int getFrames()=0;
	virtual int stop() = 0;
	virtual int showFrames() = 0;
	virtual int projectPoints(std::vector<float> &From, std::vector<float> &To) = 0;	//Function for camera/device specific capability to project points to/from pixels to real world.

};