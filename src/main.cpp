#include "PEACKTracker_OpenPose.h"
#include "PEACK_Camera_RealSense.h"
#include "PEACKTracker_OpenPose_Aux.h"
#include <iostream>
#include <sys/stat.h>


inline bool file_exists(const std::string& name) {
	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);
}

int main(int argc, char ** argv)
{

	if (argc == 2)
	{
		std::string fn = argv[1];
		
		if (file_exists(fn))
		{
			size_t lastindex = fn.find_last_of(".");
			std::string OutFile = fn.substr(0, lastindex) + ".csv";
			PEACKTracker_OpenPose Tracker1;
			PEACKDevice * cam = new PEACKrealsense;
			//cam->init(848, 480, 60);
			//cam->init("20190415_105703.bag");
			cam->init(fn);
			
			Tracker1.init(PEACKTracker_Pose, PEACKDetection_Mode_3D, cam, OutFile);
			
			
			//for (int i = 0; i < 100; i++)
			auto start = std::chrono::high_resolution_clock::now();
			
			while (Tracker1.dev->CurrentStatus != PEACK_DEVICE_STATUS_STOPPED)
			{
				Tracker1.getFrameFromDevice();
				//Tracker1.dev->showFrames();
				
				Tracker1.predictKeypoints();
				Tracker1.processKeypoints3D();
				//ProcessKeypointsForRealSense(Tracker1);
				
				//std::cout << "Device run status: " << Tracker1.dev->CurrentStatus << std::endl;
			}

			auto end = std::chrono::high_resolution_clock::now();

			std::chrono::duration<double> elapsed = end - start;
			std::cout << "Execution time: " << elapsed.count() << std::endl;
		}
		else
			std::cout << fn << " not found!" << std::endl;
	}
	else
		std::cout << "Usage: PEACK_track <filename.bag>";

	//system("pause");
	return 0;
}