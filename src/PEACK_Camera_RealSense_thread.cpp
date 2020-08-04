#include "PEACK_Camera_RealSense.h"
#include "librealsense2\rsutil.h"
#include <opencv2/dnn_superres.hpp>
#include <opencv2/dnn.hpp>

void print_intrinsics(rs2_intrinsics& intrinsics)
{
	std::cout << intrinsics.fx << '\t' << intrinsics.fy << '\t' << intrinsics.width;
	std::cout << '\t' << intrinsics.height << '\t' << intrinsics.ppx << '\t' << intrinsics.ppy;
	std::cout << std::endl;
}

static std::string get_device_name(const rs2::device& dev)
{
	// Each device provides some information on itself, such as name:
	std::string name = "Unknown Device";
	if (dev.supports(RS2_CAMERA_INFO_NAME))
		name = dev.get_info(RS2_CAMERA_INFO_NAME);

	// and the serial number of the device:
	std::string sn = "########";
	if (dev.supports(RS2_CAMERA_INFO_SERIAL_NUMBER))
		sn = std::string("#") + dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);

	return name + " " + sn;
}

int PEACKrealsense::init(int Width, int Height, int Rate)
{
	FrameCount = 0;
	rs2::context ctx;
	rs2::device_list devices = ctx.query_devices();
	Device_Type = PEACK_DEVICE_CAMERA;
	CamModel = get_device_name(devices[0]);
	std::cout << "Using device: " << CamModel << std::endl;
	CurrentStatus = PEACK_DEVICE_STATUS_RUNNING;
	
	InputFile = "-1";
	AnyFrames = 0;
	FrameWidth = Width;
	FrameHeight = Height;
	FrameRate = Rate;
	align = new rs2::align(RS2_STREAM_COLOR);
	cfg.enable_stream(RS2_STREAM_COLOR, FrameWidth, FrameHeight, RS2_FORMAT_RGB8, FrameRate);
	cfg.enable_stream(RS2_STREAM_DEPTH, FrameWidth, FrameHeight, RS2_FORMAT_Z16, FrameRate);
	//cfg.enable_record_to_file("record.bag");
	pipe.start(cfg);

	intrinsics = pipe.get_active_profile().get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_intrinsics();
	dec_filter.set_option(RS2_OPTION_FILTER_MAGNITUDE, 3);
	spat_filter.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.55f);

	rs2::frameset frames;
	for (int i = 0; i < 30; i++)
	{
		//Wait for all configured streams to produce a frame
		frames = pipe.wait_for_frames();
	}
	rcolor_frame = frames.get_color_frame();
	rdepth_frame = frames.get_depth_frame();
	rs2::frame filtered = rdepth_frame;
	filtered = dec_filter.process(filtered);
	filtered = spat_filter.process(filtered);
	rdepth_frame = filtered;
	FirstTimeStamp = rdepth_frame.get_timestamp()/1000.0;

	return 0;
}
 
int PEACKrealsense::init(std::string fn)
{
	FrameCount = 0;
	InputFile = fn;
	Device_Type = PEACK_DEVICE_FILE;
	std::cout << "Using device: Realsense Rosbag file " << InputFile << std::endl;

	AnyFrames = 0;

	align = new rs2::align(RS2_STREAM_COLOR);
	
	
	cfg.enable_device_from_file(InputFile, false);
	pipe.start(cfg);
	device = pipe.get_active_profile().get_device();
	playback = new rs2::playback(device.as<rs2::playback>());
	playback->set_real_time(false);
	rs2::frameset frames;
	
	intrinsics = pipe.get_active_profile().get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_intrinsics();
	print_intrinsics(intrinsics);
	dec_filter.set_option(RS2_OPTION_FILTER_MAGNITUDE, 3);
	spat_filter.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.55f);

	alive = true;
	
	std::cout << "Setting filter options\n";
	while(!pipe.poll_for_frames(&frames));
	CurrentStatus = PEACK_DEVICE_STATUS_RUNNING;

	FrameWidth = frames.get_depth_frame().get_width();
	FrameHeight = frames.get_depth_frame().get_height();
	FrameRate = -1;	//File
	FirstTimeStamp = frames.get_depth_frame().get_timestamp() / 1000.0;
	
	std::cout << "Frame Width: " << FrameWidth << std::endl;
	std::cout << "Frame Height: " << FrameHeight << std::endl;
	std::cout << "Intial Frame number: " << frames.get_depth_frame().get_frame_number() << std::endl;
	return 0;
}

int PEACKrealsense::getFrames()
{
	bool go = false;
	rs2::frameset frames;
	rs2::disparity_transform disparity2depth(false);
	std::thread video_processing_thread([&]() {
		// In order to generate new composite frames, we have to wrap the processing
		// code in a lambda
		rs2::processing_block frame_processor(
			[&](rs2::frameset data, // Input frameset (from the pipeline)
				rs2::frame_source& source) // Frame pool that can allocate new frames
			{
				// First make the frames spatially aligned

				// Next, apply depth post-processing
				rs2::frame depth = data.get_depth_frame();
				// Decimation will reduce the resultion of the depth image,
				// closing small holes and speeding-up the algorithm
				depth = dec_filter.process(depth);
				// To make sure far-away objects are filtered proportionally
				// we try to switch to disparity domain
				depth = depth2disparity.process(depth);
				// Apply spatial filtering
				depth = spat_filter.process(depth);
				// Apply temporal filtering
				depth = temp_filter.process(depth);
				// If we are in disparity domain, switch back to depth
				depth = disparity2depth.process(depth);

				//checking the size before align process, due to decmiation set to 2, it will redcue the size as 1/2
				float width = depth.as<rs2::video_frame>().get_width();
				float height = depth.as<rs2::video_frame>().get_height();

				auto color = data.get_color_frame();
				rs2::frameset combined = source.allocate_composite_frame({ depth, color });
				combined = align->process(combined);

				//checking the size after align process, the size align to color 640*480
				width = combined.get_depth_frame().as<rs2::video_frame>().get_width();
				height = combined.get_depth_frame().as<rs2::video_frame>().get_height();

				source.frame_ready(combined);

				
			});
		// Indicate that we want the results of frame_processor
		// to be pushed into postprocessed_frames queue
		frame_processor >> postprocessed_frames;

		if (Device_Type == PEACK_DEVICE_CAMERA)
		{
			frames = pipe.wait_for_frames();
			go = true;
		}
		else if (Device_Type == PEACK_DEVICE_FILE)
		{
			go = pipe.poll_for_frames(&frames);
			AnyFrames = go;
			std::cout << "Reached here!\n";
			if (playback->current_status() == RS2_PLAYBACK_STATUS_STOPPED)
				CurrentStatus = PEACK_DEVICE_STATUS_STOPPED;
		}

		if (frames.size() != 0) frame_processor.invoke(frames);
		
	});

	
	while (go != true)
	{
		
		video_processing_thread.join();
		static rs2::frameset frameset;
		postprocessed_frames.poll_for_frame(&frameset);
		//frameset = postprocessed_frames.wait_for_frame();
		frameset = align->process(frameset);
		rcolor_frame = frameset.get_color_frame();
		rdepth_frame = frameset.get_depth_frame();
		
		int DFrameWidth = frameset.get_depth_frame().get_width();
		int DFrameHeight = frameset.get_depth_frame().get_height();
		intrinsics = rdepth_frame.get_profile().as<rs2::video_stream_profile>().get_intrinsics();
		std::cout << "Depth frame dims: " << DFrameWidth << '\t' << DFrameHeight << std::endl;
		std::cout << "Color frame dims: " << frames.get_color_frame().get_width() << '\t' << frames.get_color_frame().get_height() << std::endl;
		
		//std::chrono::milliseconds()
		//std::this_thread::sleep_for(std::chrono::milliseconds(1000));


		TimeStamp = frameset.get_timestamp() / 1000.0 - FirstTimeStamp;
		ColorFrame = cv::Mat(cv::Size(FrameWidth, FrameHeight), CV_8UC3, (void*)rcolor_frame.get_data(), cv::Mat::AUTO_STEP);
		DepthFrame = cv::Mat(cv::Size(DFrameWidth, DFrameHeight), CV_16UC1, (void*)rdepth_frame.get_data(), cv::Mat::AUTO_STEP);

		/* if (frames.get_color_frame().get_width() != filtered.get_width())
		{
			cv::Mat Temp;
			cv::resize(ColorFrame, Temp, cv::Size(284, 160), 0, 0, cv::INTER_NEAREST);
			ColorFrame = Temp.clone();
			std::cout << "Color frame dims: " << DepthFrame.size() << std::endl;
		}*/
		AnyFrames = 1;
		//std::cout << "Frame count: " << ++FrameCount << '\t' << TimeStamp << '\t' << d.get_timestamp() / 100000 << std::endl;
		
		return 0;
	}

	std::cout << "Uh oh Reached here!\n";
	video_processing_thread.join();
	return -1;
}

int PEACKrealsense::showFrames()
{
	if (AnyFrames)
	{
		cv::namedWindow("Color Image", cv::WINDOW_AUTOSIZE);
		cv::namedWindow("Depth Image", cv::WINDOW_AUTOSIZE);

		cv::imshow("Color Image", ColorFrame);
		cv::imshow("Depth Image", DepthFrame);

		cv::waitKey(0);
	}
	return 0;
}

int PEACKrealsense::projectPoints(std::vector<float>& From, std::vector<float>& To)
{

	float Sx = double(ColorFrame.size().width) / double(DepthFrame.size().width);
	float Sy = double(ColorFrame.size().height) / double(DepthFrame.size().height);
	//std::cout << "Scale X: " << Sx << '\t' << "Scale Y: " << Sy << std::endl;
	float pixels[2] = { From[0]/Sx, From[1]/Sy };
	rs2::depth_frame d = rdepth_frame;
	float depth = d.get_distance(pixels[0], pixels[1]);
	//float pixels2[2] = { 2.986 * From[0], 3 * From[1] };
	rs2_deproject_pixel_to_point(&To[0], &intrinsics, &pixels[0], depth);
	
	return 0;
}

int PEACKrealsense::stop()
{
	pipe.stop();
	cfg.disable_all_streams();
	delete align;
	return 0;
}

/*
int main()
{
	PEACKDevice * cam = new PEACKrealsense;
	cam->init(848,480,60);
	cam->getFrames();
	cam->showFrames();
	

	return 0;
}
*/