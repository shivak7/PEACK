#include <PEACKTracker_CubeMos.h>
#include <ThirdPartyCode/samples.h>

//const std::map<unsigned int, std::string> POSE_BODY_18_BODY_PARTS{
//	{ 0,  "Nose" },
//	{ 1,  "Neck" },
//	{ 2,  "RShoulder" },
//	{ 3,  "RElbow" },
//	{ 4,  "RWrist" },
//	{ 5,  "LShoulder" },
//	{ 6,  "LElbow" },
//	{ 7,  "LWrist" },
//	{ 8,  "RHip" },
//	{ 9,  "RKnee" },
//	{ 10, "RAnkle" },
//	{ 11, "LHip" },
//	{ 12, "LKnee" },
//	{ 13, "LAnkle" },
//	{ 14, "REye" },
//	{ 15, "LEye" },
//	{ 16, "REar" },
//	{ 17, "LEar" }	
//};


static cv::Scalar const skeletonColor = cv::Scalar(100, 254, 213);
static cv::Scalar const jointColor = cv::Scalar(222, 55, 22);

typedef std::unique_ptr<CM_SKEL_Buffer, void (*)(CM_SKEL_Buffer*)> CUBEMOS_SKEL_Buffer_Ptr;

CUBEMOS_SKEL_Buffer_Ptr
create_skel_buffer()
{
	return CUBEMOS_SKEL_Buffer_Ptr(new CM_SKEL_Buffer(), [](CM_SKEL_Buffer* pb) {
		cm_skel_release_buffer(pb);
		delete pb;
		});
}

/*
Render skeletons and tracking ids on top of the color image
*/
inline void
renderSkeletons2D(const CM_SKEL_Buffer* skeletons_buffer, cv::Mat& image)
{
    CV_Assert(image.type() == CV_8UC3);
    const cv::Point2f absentKeypoint(-1.0f, -1.0f);

    const std::vector<std::pair<int, int>> limbKeypointsIds = { { 1, 2 },   { 1, 5 },   { 2, 3 }, { 3, 4 },  { 5, 6 },
                                                                { 6, 7 },   { 1, 8 },   { 8, 9 }, { 9, 10 }, { 1, 11 },
                                                                { 11, 12 }, { 12, 13 }, { 1, 0 }, { 0, 14 }, { 14, 16 },
                                                                { 0, 15 },  { 15, 17 } };

    for (int i = 0; i < skeletons_buffer->numSkeletons; i++) {
        CV_Assert(skeletons_buffer->skeletons[i].numKeyPoints == 18);

        int id = skeletons_buffer->skeletons[i].id;
        cv::Point2f keyPointHead(skeletons_buffer->skeletons[i].keypoints_coord_x[0],
            skeletons_buffer->skeletons[i].keypoints_coord_y[0]);

        for (size_t keypointIdx = 0; keypointIdx < skeletons_buffer->skeletons[i].numKeyPoints; keypointIdx++) {
            const cv::Point2f keyPoint(skeletons_buffer->skeletons[i].keypoints_coord_x[keypointIdx],
                skeletons_buffer->skeletons[i].keypoints_coord_y[keypointIdx]);
            if (keyPoint != absentKeypoint) {
                cv::circle(image, keyPoint, 4, jointColor, -1);
            }
        }

        for (const auto& limbKeypointsId : limbKeypointsIds) {
            const cv::Point2f keyPointFirst(skeletons_buffer->skeletons[i].keypoints_coord_x[limbKeypointsId.first],
                skeletons_buffer->skeletons[i].keypoints_coord_y[limbKeypointsId.first]);

            const cv::Point2f keyPointSecond(skeletons_buffer->skeletons[i].keypoints_coord_x[limbKeypointsId.second],
                skeletons_buffer->skeletons[i].keypoints_coord_y[limbKeypointsId.second]);

            if (keyPointFirst == absentKeypoint || keyPointSecond == absentKeypoint) {
                continue;
            }

            cv::line(image, keyPointFirst, keyPointSecond, skeletonColor, 2, cv::LINE_AA);
        }
        for (size_t keypointIdx = 0; keypointIdx < skeletons_buffer->skeletons[i].numKeyPoints; keypointIdx++) {
            const cv::Point2f keyPoint(skeletons_buffer->skeletons[i].keypoints_coord_x[keypointIdx],
                skeletons_buffer->skeletons[i].keypoints_coord_y[keypointIdx]);
            if (keyPoint != absentKeypoint) {
                // found a valid keypoint and displaying the skeleton tracking id next to it
                cv::putText(image,
                    (std::to_string(id)),
                    cv::Point2f(keyPoint.x, keyPoint.y - 20),
                    cv::FONT_HERSHEY_COMPLEX,
                    2,
                    skeletonColor);
                break;
            }
        }
    }

	//cv::imshow("Render", image);
	//cv::waitKey(1);
}

int PEACKTracker_CubeMos::populateMap()
{
	SkeletonMap.insert(PEACK_JointMap::value_type("Nose", 0));
	SkeletonMap.insert(PEACK_JointMap::value_type("Neck", 1));
	SkeletonMap.insert(PEACK_JointMap::value_type("RShoulder", 2));
	SkeletonMap.insert(PEACK_JointMap::value_type("RElbow", 3));
	SkeletonMap.insert(PEACK_JointMap::value_type("RWrist", 4));
	SkeletonMap.insert(PEACK_JointMap::value_type("LShoulder", 5));
	SkeletonMap.insert(PEACK_JointMap::value_type("LElbow", 6));
	SkeletonMap.insert(PEACK_JointMap::value_type("LWrist", 7));
	SkeletonMap.insert(PEACK_JointMap::value_type("RHip", 8));
	SkeletonMap.insert(PEACK_JointMap::value_type("RKnee", 9));
	SkeletonMap.insert(PEACK_JointMap::value_type("RAnkle", 10));
	SkeletonMap.insert(PEACK_JointMap::value_type("LHip", 11));
	SkeletonMap.insert(PEACK_JointMap::value_type("LKnee", 12));
	SkeletonMap.insert(PEACK_JointMap::value_type("LAnkle", 13));
	SkeletonMap.insert(PEACK_JointMap::value_type("REye", 14));
	SkeletonMap.insert(PEACK_JointMap::value_type("LEye", 15));
	SkeletonMap.insert(PEACK_JointMap::value_type("REar", 16));
	SkeletonMap.insert(PEACK_JointMap::value_type("LEar", 17));
}
int PEACKTracker_CubeMos::getMappingFromString(std::string str)
{
	return SkeletonMap.left.find(str)->second;
}

std::string PEACKTracker_CubeMos::getMappingFromIndex(unsigned int idx)
{
	return SkeletonMap.right.find(idx)->second;
}

int PEACKTracker_CubeMos::init(int target, int mode, PEACKDevice* device, std::string fn)
{
	populateMap();
	dev = device;
	OutFile.open(fn);
	TrackerName = "OpenPose";
	TrackerTarget = target;
	TrackerMode = mode;
	FirstDetection = true;
	video.open("Demo.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(848, 480));

	enInferenceMode = CM_TargetComputeDevice::CM_CPU;
	handle = nullptr;
	cm_initialise_logging(CM_LogLevel::CM_LL_INFO, true, default_log_dir().c_str());

	char* tmpvar;
	size_t len;
	errno_t err = _dupenv_s(&tmpvar, &len, "LOCALAPPDATA");
	std::string lic = "\\Cubemos\\SkeletonTracking\\license";
	lic = tmpvar + lic;
	std::cout << "License: " << tmpvar << std::endl;
	retCode = cm_skel_create_handle(&handle, lic.c_str());
	CHECK_HANDLE_CREATION(retCode);

	std::string modelName = default_model_dir();
	if (enInferenceMode == CM_TargetComputeDevice::CM_CPU) {
		modelName += std::string("fp32/skeleton-tracking.cubemos");
	}
	else {
		modelName += std::string("fp16/skeleton-tracking.cubemos");
	}
	retCode = cm_skel_load_model(handle, enInferenceMode, modelName.c_str());
	if (retCode != CM_SUCCESS) {
		EXIT_PROGRAM("Model loading failed.");
	}

	skeletRequestHandle = nullptr;
	cm_skel_create_async_request_handle(handle, &skeletRequestHandle);
	dev->getFrames();
	imageLast = {
		dev->ColorFrame.data,         CM_UINT8, dev->ColorFrame.cols, dev->ColorFrame.rows, dev->ColorFrame.channels(),
		(int)dev->ColorFrame.step[0], CM_HWC
	};
	
	//skeletonsPresent = create_skel_buffer();
	//skeletonsLast = create_skel_buffer();

	nTimeoutMs = 1000;

	CM_ReturnCode retCodeFirstFrame =
		cm_skel_estimate_keypoints_start_async(handle, skeletRequestHandle, &imageLast, nHeight);
	retCodeFirstFrame = cm_skel_wait_for_keypoints(handle, skeletRequestHandle, skeletonsLast.get(), nTimeoutMs);

}

PEACKTracker_CubeMos::~PEACKTracker_CubeMos()
{
	std::cout << "Destructor called!" << std::endl;
	OutFile << FileBuffer.str();
	OutFile.close();
	dev->stop();
	video.release();
	cm_skel_destroy_async_request_handle(&skeletRequestHandle);
	cm_skel_destroy_handle(&handle);
	delete dev;
}

int PEACKTracker_CubeMos::release()
{
	delete this;
	return 0;
}

int PEACKTracker_CubeMos::getFrameFromDevice()
{
	dev->getFrames();
	TimeStamp = dev->TimeStamp;
	//dev->TimeStamp = op::getTimeSeconds(opTimer);	//Manual overide of device timer (will not work when loading from files). Use camera's hw timestamps whenever possible
	return 0;
}

int PEACKTracker_CubeMos::predictKeypoints()
{
	// Free memory of the latest frame
	cm_skel_release_buffer(skeletonsPresent.get());

	if (TrackerMode == PEACKDetection_Mode_2D)
	{
		imageLast = {
		dev->ColorFrame.data,         CM_UINT8, dev->ColorFrame.cols, dev->ColorFrame.rows, dev->ColorFrame.channels(),
		(int)dev->ColorFrame.step[0], CM_HWC
		};

		retCode = cm_skel_estimate_keypoints_start_async(handle, skeletRequestHandle, &imageLast, nHeight);
		retCode = cm_skel_wait_for_keypoints(handle, skeletRequestHandle, skeletonsPresent.get(), nTimeoutMs);

		if (retCode == CM_SUCCESS) {
			if (skeletonsPresent->numSkeletons > 0) {
				if (dev->AnyFrames) {
					// Assign tracking ids to the skeletons in the present frame
					cm_skel_update_tracking_id(handle, skeletonsLast.get(), skeletonsPresent.get());
					// Render skeleton overlays with tracking ids
					renderSkeletons2D(skeletonsPresent.get(), dev->ColorFrame);
					int ch = cv::waitKey(1);
					if (ch == 27)
						dev->CurrentStatus = PEACK_DEVICE_STATUS_STOPPED;
					// Set the present frame as last one to track the next frame
					skeletonsLast.swap(skeletonsPresent);

					dev->AnyFrames = 0;
					NumBodyParts = skeletonsPresent.get()->skeletons[0].numKeyPoints;
				}
			}
		}
	}

	return 0;
}

int PEACKTracker_CubeMos::getPartKeyPoint(int part_idx, std::vector<float>& V)
{
	if (TrackerMode == PEACKDetection_Mode_2D)
	{
		int person = 0;
		auto x = skeletonsPresent.get()->skeletons[person].keypoints_coord_x[part_idx];
		auto y = skeletonsPresent.get()->skeletons[person].keypoints_coord_y[part_idx];
		auto z = 0;
		auto score = skeletonsPresent.get()->skeletons[person].confidences[part_idx];

		V.push_back(x);
		V.push_back(y);
		V.push_back(z);
		V.push_back(score);

		return 0;
	}

}

int PEACKTracker_CubeMos::processKeypoints2D()
{
	std::vector<float> From;
	
	if (FirstDetection == true)
	{
		FileBuffer << "Time" << ",";
		for (int part_idx = 0; part_idx < SkeletonMap.size(); part_idx++)
		{
			std::string Part = getMappingFromIndex(part_idx);
			std::string X_string = Part + " X";
			std::string Y_string = Part + " Y";
			std::string Z_string = Part + " Z";
			FileBuffer << X_string << "," << Y_string << "," << Z_string; //<< "," << score;
			if (part_idx < SkeletonMap.size() - 1)
				FileBuffer << ",";
		}
		FileBuffer << std::endl;
		FirstDetection = false;
	}

	//std::cout << "Time:" << dev->TimeStamp << std::endl;
	//std::cout << "NumBodyParts:" << NumBodyParts << std::endl;
	FileBuffer << TimeStamp << ",";
	for (int part_idx = 0; part_idx < NumBodyParts; part_idx++)
	{
		From.clear();
		getPartKeyPoint(part_idx, From);

		//std::cout << "Body part:" << part_idx + 1 << '\t' << From[0] << '\t' << From[1] << '\t' << From[2] << std::endl;
		FileBuffer << From[0] << "," << From[1] << "," << From[2];
		if (part_idx < NumBodyParts - 1)
			FileBuffer << ",";
	}
	FileBuffer << std::endl;
	return 0;
}

int PEACKTracker_CubeMos::processKeypoints3D()
{
	std::vector<float> From;
	std::vector<float> To(3, 0);
	if (FirstDetection == true)
	{
		FileBuffer << "Time" << ",";
		for (int part_idx = 0; part_idx < SkeletonMap.size(); part_idx++)
		{
			std::string Part = getMappingFromIndex(part_idx);
			std::string X_string = Part + " X";
			std::string Y_string = Part + " Y";
			std::string Z_string = Part + " Z";
			FileBuffer << X_string << "," << Y_string << "," << Z_string; //<< "," << score;
			if (part_idx < SkeletonMap.size() - 1)
				FileBuffer << ",";
		}
		FileBuffer << std::endl;
		FirstDetection = false;
	}

	//std::cout << "Time:" << dev->TimeStamp << std::endl;
	FileBuffer << TimeStamp << ",";
	for (int part_idx = 0; part_idx < NumBodyParts; part_idx++)
	{
		From.clear();

		getPartKeyPoint(part_idx, From);
		dev->projectPoints(From, To);

		//std::cout << "Body part:" << part_idx + 1 << '\t' << To[0] << '\t' << To[1] << '\t' << To[2] << std::endl;
		FileBuffer << To[0] << "," << To[1] << "," << To[2];
		if (part_idx < NumBodyParts - 1)
			FileBuffer << ",";
	}
	FileBuffer << std::endl;
	return 0;
}