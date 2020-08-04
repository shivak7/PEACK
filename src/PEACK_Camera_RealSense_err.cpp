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
	
	dec_filter.set_option(RS2_OPTION_FILTER_MAGNITUDE, 3);
	spat_filter.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.55f);
	
	std::cout << "Setting filter options\n";
	while(!pipe.poll_for_frames(&frames));
	CurrentStatus = PEACK_DEVICE_STATUS_RUNNING;

	intrinsics = pipe.get_active_profile().get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_intrinsics();
	print_intrinsics(intrinsics);


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
	rs2::frameset frames;
	bool go = false;
	if (Device_Type == PEACK_DEVICE_CAMERA)
	{
		frames = pipe.wait_for_frames();
		go = true;
	}
	else if (Device_Type == PEACK_DEVICE_FILE)
	{
		go = pipe.poll_for_frames(&frames);
		AnyFrames = go;
		if (playback->current_status() == RS2_PLAYBACK_STATUS_STOPPED)
			CurrentStatus = PEACK_DEVICE_STATUS_STOPPED;
	}

	if (go == true)
	{
		frames = align->process(frames);
		rcolor_frame = frames.get_color_frame();
		rdepth_frame = frames.get_depth_frame();

		//rs2::depth_frame filtered = rdepth_frame;
		//filtered = dec_filter.process(filtered);
		//filtered = spat_filter.process(filtered);
		//rdepth_frame = filtered;
		intrinsics = rdepth_frame.get_profile().as<rs2::video_stream_profile>().get_intrinsics();
		print_intrinsics(intrinsics);
		exit(-1);
		//std::cout << "Depth frame dims: " << filtered.get_width() << '\t' << filtered.get_height() << std::endl;
		//std::cout << "Color frame dims: " << frames.get_color_frame().get_width() << '\t' << frames.get_color_frame().get_height() << std::endl;
		
		
		//std::chrono::milliseconds()
		//std::this_thread::sleep_for(std::chrono::milliseconds(1000));


		TimeStamp = frames.get_timestamp() / 1000.0 - FirstTimeStamp;
		
		//std::cout << "Time:" << TimeStamp << std::endl;
		rs2::depth_frame d = rdepth_frame;
		//std::cout << "Depth Frame:" << d.get_frame_number() << std::endl;
		/*
		cv::dnn_superres::DnnSuperResImpl sr;
		std::string model_str = "EDSR_x2.pb";
		sr.readModel(model_str);
		sr.setModel("edsr", 2);
		*/
		ColorFrame = cv::Mat(cv::Size(FrameWidth, FrameHeight), CV_8UC3, (void*)rcolor_frame.get_data(), cv::Mat::AUTO_STEP);
		//cv::Mat temp = ColorFrame.clone();
		//sr.upsample(temp, ColorFrame);
		//cv::fastNlMeansDenoisingColored(temp, ColorFrame, 10, 10, 7, 21);
		
		DepthFrame = cv::Mat(cv::Size(FrameWidth, FrameHeight), CV_16UC1, (void*)rdepth_frame.get_data(), cv::Mat::AUTO_STEP);

		AnyFrames = 1;
		//std::cout << "Frame count: " << ++FrameCount << '\t' << TimeStamp << '\t' << d.get_timestamp() / 100000 << std::endl;
		return 0;
	}

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

	float pixels[2] = { From[0], From[1] };
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