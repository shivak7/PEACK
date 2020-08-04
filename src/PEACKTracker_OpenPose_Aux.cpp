#include "PEACKTracker_OpenPose_Aux.h"

extern const std::map<unsigned int, std::string> POSE_BODY_25_BODY_PARTS{
	{ 0,  "Nose" },
	{ 1,  "Neck" },
	{ 2,  "RShoulder" },
	{ 3,  "RElbow" },
	{ 4,  "RWrist" },
	{ 5,  "LShoulder" },
	{ 6,  "LElbow" },
	{ 7,  "LWrist" },
	{ 8,  "MidHip" },
	{ 9,  "RHip" },
	{ 10, "RKnee" },
	{ 11, "RAnkle" },
	{ 12, "LHip" },
	{ 13, "LKnee" },
	{ 14, "LAnkle" },
	{ 15, "REye" },
	{ 16, "LEye" },
	{ 17, "REar" },
	{ 18, "LEar" },
	{ 19, "LBigToe" },
	{ 20, "LSmallToe" },
	{ 21, "LHeel" },
	{ 22, "RBigToe" },
	{ 23, "RSmallToe" },
	{ 24, "RHeel" },
	{ 25, "Background" }
};

int ProcessKeypointsForRealSense(PEACKTracker & T)
{
	std::vector<float> From;
	std::vector<float> To(3, 0);
	if (T.FirstDetection == true)
	{
		T.FileBuffer << "Time" << ",";
		for (int part_idx = 0; part_idx < POSE_BODY_25_BODY_PARTS.size()-1; part_idx++)
		{
			std::string Part = POSE_BODY_25_BODY_PARTS.find(part_idx)->second;
			std::string X_string = Part + " X";
			std::string Y_string = Part + " Y";
			std::string Z_string = Part + " Z";
			T.FileBuffer << X_string << "," << Y_string << "," << Z_string; //<< "," << score;
			if (part_idx < POSE_BODY_25_BODY_PARTS.size() - 2)
				T.FileBuffer << ",";
		}
		T.FileBuffer << std::endl;
		T.FirstDetection = false;
	}

	//std::cout << "Time:" << T.dev->TimeStamp << std::endl;
	T.FileBuffer << T.TimeStamp << ",";
	for (int part_idx = 0; part_idx < T.NumBodyParts; part_idx++)
	{
		From.clear();

		T.getPartKeyPoint(part_idx, From);
		T.dev->projectPoints(From, To);

		//std::cout << "Body part:" << part_idx + 1 << '\t' << To[0] << '\t' << To[1] << '\t' << To[2] << std::endl;
		T.FileBuffer << To[0] << "," << To[1] << "," << To[2];
		if (part_idx < T.NumBodyParts - 1)
			T.FileBuffer << ",";
	}
	T.FileBuffer << std::endl;
	return 0;
}
