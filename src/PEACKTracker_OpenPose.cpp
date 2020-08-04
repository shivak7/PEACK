#include "PEACKTracker_OpenPose.h"
#include "PEACK_Camera_RealSense.h"
#include <openpose/flags.hpp>

const std::map<unsigned int, std::string> POSE_BODY_25_BODY_PARTS{
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

void printKeypoints(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
{
	try
	{
		
		// Example: How to use the pose keypoints
		if (datumsPtr != nullptr && !datumsPtr->empty())
		{
			op::opLog("Body keypoints: " + datumsPtr->at(0)->poseKeypoints.toString(), op::Priority::High);
			op::opLog("Face keypoints: " + datumsPtr->at(0)->faceKeypoints.toString(), op::Priority::High);
			op::opLog("Left hand keypoints: " + datumsPtr->at(0)->handKeypoints[0].toString(), op::Priority::High);
			op::opLog("Right hand keypoints: " + datumsPtr->at(0)->handKeypoints[1].toString(), op::Priority::High);
		}
		else
			op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High);
	}
	catch (const std::exception& e)
	{
		op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
	}
}

void display(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
{
	try
	{
		
		// User's displaying/saving/other processing here
		// datum.cvOutputData: rendered frame with pose or heatmaps
		// datum.poseKeypoints: Array<float> with the estimated pose
		if (datumsPtr != nullptr && !datumsPtr->empty())
		{
			// Display image
			const cv::Mat cvMat = OP_OP2CVCONSTMAT(datumsPtr->at(0)->cvOutputData);

			if (!cvMat.empty())
			{
				cv::imshow(OPEN_POSE_NAME_AND_VERSION + " - Tutorial C++ API", cvMat);
				cv::waitKey(1);
			}
			else
				op::opLog("Empty cv::Mat as output.", op::Priority::High, __LINE__, __FUNCTION__, __FILE__);
			
		}
		else
			op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High);
	}
	catch (const std::exception& e)
	{
		op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
	}
}

void configureWrapper(op::Wrapper& opWrapper)
{
	try
	{
		// Configuring OpenPose

		// logging_level
		op::checkBool(
			0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
			__LINE__, __FUNCTION__, __FILE__);
		op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
		op::Profiler::setDefaultX(FLAGS_profile_speed);

		// Applying user defined configuration - GFlags to program variables
		// outputSize
		const auto outputSize = op::flagsToPoint(op::String(FLAGS_output_resolution), "-1x-1");
		// netInputSize
		const auto netInputSize = op::flagsToPoint(op::String(FLAGS_net_resolution), "-1x368");
		// faceNetInputSize
		const auto faceNetInputSize = op::flagsToPoint(op::String(FLAGS_face_net_resolution), "368x368 (multiples of 16)");
		// handNetInputSize
		const auto handNetInputSize = op::flagsToPoint(op::String(FLAGS_hand_net_resolution), "368x368 (multiples of 16)");
		// poseMode
		const auto poseMode = op::flagsToPoseMode(FLAGS_body);
		// poseModel
		const auto poseModel = op::flagsToPoseModel(op::String(FLAGS_model_pose));
		// JSON saving
		if (!FLAGS_write_keypoint.empty())
			op::opLog(
				"Flag `write_keypoint` is deprecated and will eventually be removed. Please, use `write_json`"
				" instead.", op::Priority::Max);
		// keypointScaleMode
		const auto keypointScaleMode = op::flagsToScaleMode(FLAGS_keypoint_scale);
		// heatmaps to add
		const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg,
			FLAGS_heatmaps_add_PAFs);
		const auto heatMapScaleMode = op::flagsToHeatMapScaleMode(FLAGS_heatmaps_scale);
		// >1 camera view?
		const auto multipleView = (FLAGS_3d || FLAGS_3d_views > 1);
		// Face and hand detectors
		const auto faceDetector = op::flagsToDetector(FLAGS_face_detector);
		const auto handDetector = op::flagsToDetector(FLAGS_hand_detector);
		// Enabling Google Logging
		const bool enableGoogleLogging = true;

		// Pose configuration (use WrapperStructPose{} for default and recommended configuration)
		const op::WrapperStructPose wrapperStructPose{
			poseMode, netInputSize, outputSize, keypointScaleMode, FLAGS_num_gpu, FLAGS_num_gpu_start,
			FLAGS_scale_number, (float)FLAGS_scale_gap, op::flagsToRenderMode(FLAGS_render_pose, multipleView),
			poseModel, !FLAGS_disable_blending, (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap,
			FLAGS_part_to_show, op::String(FLAGS_model_folder), heatMapTypes, heatMapScaleMode, FLAGS_part_candidates,
			(float)FLAGS_render_threshold, FLAGS_number_people_max, FLAGS_maximize_positives, FLAGS_fps_max,
			op::String(FLAGS_prototxt_path), op::String(FLAGS_caffemodel_path),
			(float)FLAGS_upsampling_ratio, enableGoogleLogging };
		opWrapper.configure(wrapperStructPose);
		// Face configuration (use op::WrapperStructFace{} to disable it)
		const op::WrapperStructFace wrapperStructFace{
			FLAGS_face, faceDetector, faceNetInputSize,
			op::flagsToRenderMode(FLAGS_face_render, multipleView, FLAGS_render_pose),
			(float)FLAGS_face_alpha_pose, (float)FLAGS_face_alpha_heatmap, (float)FLAGS_face_render_threshold };
		opWrapper.configure(wrapperStructFace);
		// Hand configuration (use op::WrapperStructHand{} to disable it)
		const op::WrapperStructHand wrapperStructHand{
			FLAGS_hand, handDetector, handNetInputSize, FLAGS_hand_scale_number, (float)FLAGS_hand_scale_range,
			op::flagsToRenderMode(FLAGS_hand_render, multipleView, FLAGS_render_pose), (float)FLAGS_hand_alpha_pose,
			(float)FLAGS_hand_alpha_heatmap, (float)FLAGS_hand_render_threshold };
		opWrapper.configure(wrapperStructHand);
		// Extra functionality configuration (use op::WrapperStructExtra{} to disable it)
		const op::WrapperStructExtra wrapperStructExtra{
			FLAGS_3d, FLAGS_3d_min_views, FLAGS_identification, FLAGS_tracking, FLAGS_ik_threads };
		opWrapper.configure(wrapperStructExtra);
		// Output (comment or use default argument to disable any output)
		const op::WrapperStructOutput wrapperStructOutput{
			FLAGS_cli_verbose, op::String(FLAGS_write_keypoint), op::stringToDataFormat(FLAGS_write_keypoint_format),
			op::String(FLAGS_write_json), op::String(FLAGS_write_coco_json), FLAGS_write_coco_json_variants,
			FLAGS_write_coco_json_variant, op::String(FLAGS_write_images), op::String(FLAGS_write_images_format),
			op::String(FLAGS_write_video), FLAGS_write_video_fps, FLAGS_write_video_with_audio,
			op::String(FLAGS_write_heatmaps), op::String(FLAGS_write_heatmaps_format), op::String(FLAGS_write_video_3d),
			op::String(FLAGS_write_video_adam), op::String(FLAGS_write_bvh), op::String(FLAGS_udp_host),
			op::String(FLAGS_udp_port) };
		opWrapper.configure(wrapperStructOutput);
		// No GUI. Equivalent to: opWrapper.configure(op::WrapperStructGui{});
		// Set to single-thread (for sequential processing and/or debugging and/or reducing latency)
		if (FLAGS_disable_multi_thread)
			opWrapper.disableMultiThreading();
	}
	catch (const std::exception& e)
	{
		op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
	}
}

int PEACKTracker_OpenPose::init(int target, int mode, PEACKDevice * device, std::string fn)
{
	dev = device;
	OutFile.open(fn);
	TrackerName = "OpenPose";
	TrackerTarget = target;
	TrackerMode = mode;
	FirstDetection = true;
	video.open("Demo.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(848, 480));

	try
	{
		opTimer = op::getTimerInit();
		
		// Configuring OpenPose
		op::opLog("OpenPose: Configuring tracker...", op::Priority::High);
		configureWrapper(opWrapper);
		op::opLog("OpenPose: Starting thread(s)...", op::Priority::High);
		opWrapper.start();
		
		return 0;
	}
	catch (const std::exception& e)
	{
		return -1;
	}
	
}

PEACKTracker_OpenPose::~PEACKTracker_OpenPose()
{
	OutFile << FileBuffer.str();
	OutFile.close();
	dev->stop();
	video.release();
	delete dev;
}

int PEACKTracker_OpenPose::getFrameFromDevice()
{
	dev->getFrames();
	TimeStamp = dev->TimeStamp;
	//dev->TimeStamp = op::getTimeSeconds(opTimer);	//Manual overide of device timer (will not work when loading from files). Use camera's hw timestamps whenever possible
	return 0;
}

int PEACKTracker_OpenPose::predictKeypoints()
{
	if (dev->AnyFrames)
	try {
			const op::Matrix imageToProcess = OP_CV2OPCONSTMAT(dev->ColorFrame);
			datumProcessed = opWrapper.emplaceAndPop(imageToProcess);
			display(datumProcessed);
			const cv::Mat cvMat = OP_OP2CVCONSTMAT(datumProcessed->at(0)->cvOutputData);
			video.write(cvMat);
			dev->AnyFrames = 0;
			NumBodyParts = datumProcessed->at(0)->poseKeypoints.getSize(1);
			
			return 0;
	}
	catch (const std::exception& e)
	{
		return -1;
	}
	
}

int PEACKTracker_OpenPose::getPartKeyPoint(int part_idx, std::vector<float> &V)
{
	int person = 0;
	auto x = datumProcessed->at(0)->poseKeypoints[{person, part_idx, 0}];
	auto y = datumProcessed->at(0)->poseKeypoints[{person, part_idx, 1}];
	auto z = 0;
	auto score = datumProcessed->at(0)->poseKeypoints[{person, part_idx, 2}];

	V.push_back(x); // / 2.986);
	V.push_back(y); // / 3);
	V.push_back(z);
	V.push_back(score);
	
	return 0;
}

int PEACKTracker_OpenPose::processKeypoints2D()
{
	if (datumProcessed != nullptr)
	{
		const auto numberBodyParts = datumProcessed->at(0)->poseKeypoints.getSize(1);
		int person = 0; //Always track the first person
		if (FirstDetection==true)
		{
			
			for (int part_idx = 0; part_idx < numberBodyParts; part_idx++)
			{
				std::string Part = POSE_BODY_25_BODY_PARTS.find(part_idx)->second;
				std::string X_string = Part + " X";
				std::string Y_string = Part + " Y";
				std::string Z_string = Part + " Z";
				FileBuffer << X_string << "," << Y_string << "," << Z_string; //<< "," << score;
				if (part_idx < numberBodyParts - 1)
					FileBuffer << ",";
			}
			FileBuffer << std::endl;
			FirstDetection = false;
		}
		for (int part_idx = 0; part_idx < numberBodyParts; part_idx++)
		{
			const auto x = datumProcessed->at(0)->poseKeypoints[{person, part_idx, 0}];
			const auto y = datumProcessed->at(0)->poseKeypoints[{person, part_idx, 1}];
			const auto z = 0;
			const auto score = datumProcessed->at(0)->poseKeypoints[{person, part_idx, 2}];

			FileBuffer << TimeStamp << "," << x << "," << y << "," << z; //<< "," << score;
			if (part_idx < numberBodyParts - 1)
				FileBuffer << ",";
		}
		FileBuffer << std::endl;
		
		datumProcessed = nullptr;
	}
	else
		op::opLog("Image could not be processed.", op::Priority::High);
}

int PEACKTracker_OpenPose::processKeypoints3D()
{
	std::vector<float> From;
	std::vector<float> To(3, 0);
	if (FirstDetection == true)
	{
		FileBuffer << "Time" << ",";
		for (int part_idx = 0; part_idx < POSE_BODY_25_BODY_PARTS.size() - 1; part_idx++)
		{
			std::string Part = POSE_BODY_25_BODY_PARTS.find(part_idx)->second;
			std::string X_string = Part + " X";
			std::string Y_string = Part + " Y";
			std::string Z_string = Part + " Z";
			FileBuffer << X_string << "," << Y_string << "," << Z_string; //<< "," << score;
			if (part_idx < POSE_BODY_25_BODY_PARTS.size() - 2)
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