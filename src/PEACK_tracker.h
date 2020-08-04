#pragma once
#include <string>
#include <sstream>
#include <fstream>
#include <opencv2\opencv.hpp>
#include "PEACK_Device.h"
#include <boost/bimap.hpp>

#define PEACKTracker_Pose 1
#define PEACKTracker_Face 2
#define PEACKTracker_LeftHand 4
#define PEACKTracker_RightHand 8

#define PEACKDetection_Mode_2D 1	//Pixels
#define PEACKDetection_Mode_3D 2	//Real world coords

typedef boost::bimap<std::string, int> PEACK_JointMap;

class PEACKTracker {
public:
	std::string TrackerName;
	std::ofstream OutFile;
	std::stringstream FileBuffer;
	int TrackerMode;
	int TrackerTarget;
	int FrameWidth;
	int FrameHeight;
	int NumBodyParts;
	int DetectionMode;
	bool FirstDetection;
	double TimeStamp;
	//virtual ~PEACKTracker() { std::cout << "Base destructor called" << std::endl; };
	virtual int init(int target, int mode, PEACKDevice *, std::string FileName) = 0;
	virtual int release() = 0;
	virtual int getFrameFromDevice() = 0;
	virtual int predictKeypoints() = 0;
	virtual int processKeypoints2D() = 0;	// Process 
	virtual int processKeypoints3D() = 0;	// Process 
	virtual int getPartKeyPoint(int, std::vector<float>&) = 0; //get individual part coordinates
	virtual int populateMap() = 0;
	virtual int getMappingFromString(std::string) = 0;
	virtual std::string getMappingFromIndex(unsigned int) = 0;


//protected:
	PEACKDevice * dev;
protected:
	PEACK_JointMap SkeletonMap;

};
