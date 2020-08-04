#pragma once
#include <PEACK_Camera_generic.h>
#include <chrono>

std::string PEACKGenCam::get_device_name()
{
	std::string val = "Generic 0";
	return 0;
}

int PEACKGenCam::init(int Width, int Height, int Rate)
{

	Source.open(0);
	Source.set(cv::CAP_PROP_FRAME_WIDTH, Width);
	Source.set(cv::CAP_PROP_FRAME_HEIGHT, Height);
	CurrentStatus = PEACK_DEVICE_STATUS_RUNNING;

	InputFile = "-1";
	AnyFrames = 0;
	FrameWidth = Width;
	FrameHeight = Height;
	FrameRate = Rate;
	
	startTime = std::chrono::system_clock::now();
	return 0;
}

int PEACKGenCam::init(std::string fn)
{
	Source.open(fn);
	CurrentStatus = PEACK_DEVICE_STATUS_RUNNING;
	
	AnyFrames = 0;
	return 0;
}

int PEACKGenCam::getFrames()
{
	Source.read(ColorFrame);
	if (!ColorFrame.empty())
		AnyFrames = true;

	TimeStamp =	std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - startTime)
		.count()/1000.0;
	return 0;
}

int PEACKGenCam::stop()
{
	Source.release();
	return 0;
}

int PEACKGenCam::showFrames()
{
	if (AnyFrames)
	{
		cv::namedWindow("Color Image", cv::WINDOW_AUTOSIZE);
		cv::imshow("Color Image", ColorFrame);
		cv::waitKey(16);
	}
	return 0;
}

int PEACKGenCam::projectPoints(std::vector<float>& From, std::vector<float>& To)
{
	std::cout << "Error: This Camera does not support 3D tracking!\n";
	exit(-1);
}