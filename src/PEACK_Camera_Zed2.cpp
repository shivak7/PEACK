#include <PEACK_Camera_Zed2.h>


static cv::Mat slMat2cvMat(sl::Mat& input) 
{
    int cv_type = -1;
    switch (input.getDataType()) {
    case sl::MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
    case sl::MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
    case sl::MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
    case sl::MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
    case sl::MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
    case sl::MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
    case sl::MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
    case sl::MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
    default: break;
    }
    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(sl::MEM::CPU));
}

std::string PEACKzed2 :: get_device_name()
{

    auto camera_infos = zed2.getCameraInformation();
    //printf("Hello! This is my serial number: %d\n", camera_infos.serial_number);
    std::string name = "Zed 2";
    std::string sn;
    std::stringstream f;
    f << camera_infos.serial_number;
    f >> sn;
    
    return name + " " + sn;
}

int PEACKzed2::init(int Width, int Height, int Rate)
{
    init_parameters.camera_resolution = sl::RESOLUTION::HD720;
    init_parameters.camera_fps = Rate;
    AnyFrames = false;
    FrameCount = 0;
    //Start the camera
    sl::ERROR_CODE err = zed2.open(init_parameters);
    if (err != sl::ERROR_CODE::SUCCESS)
        exit(-1);

    return 0;
}

int PEACKzed2::init(std::string fn)
{
    FrameCount = 0;
    InputFile = fn;
    Device_Type = PEACK_DEVICE_FILE;
    std::cout << "Using device: Zed2 SVO file " << InputFile << std::endl;
    init_parameters.input.setFromSVOFile(fn.c_str());
    init_parameters.camera_disable_self_calib = true;
    init_parameters.depth_mode = sl::DEPTH_MODE::QUALITY;
    init_parameters.coordinate_units = sl::UNIT::MILLIMETER;
    AnyFrames = false;
    FrameCount = 0;
    //Start the camera
    sl::ERROR_CODE err = zed2.open(init_parameters);
    if (err != sl::ERROR_CODE::SUCCESS)
        exit(-1);
    auto resolution = zed2.getCameraInformation().camera_configuration.resolution;
    zcolor_frame.alloc(resolution, sl::MAT_TYPE::U8_C4, sl::MEM::CPU);
    zdepth_frame.alloc(resolution, sl::MAT_TYPE::U8_C4, sl::MEM::CPU);

    ColorFrame = slMat2cvMat(zcolor_frame);
    DepthFrame = slMat2cvMat(zdepth_frame);

    return 0;
}

int PEACKzed2::getFrames()
{
    
    if (zed2.grab() == sl::ERROR_CODE::SUCCESS)
    {
        zed2.retrieveImage(zcolor_frame, sl::VIEW::LEFT); // Retrieve left image
        zed2.retrieveImage(zdepth_frame, sl::VIEW::DEPTH); // Retrieve depth
        zed2.retrieveMeasure(zpoint_cloud, sl::MEASURE::XYZABGR);
        AnyFrames = true;
        return 0;
    }
    return -1;
    
}

int PEACKzed2::stop()
{
    zed2.close();
    return 0;
}


int PEACKzed2::showFrames()
{
    if (AnyFrames)
    {
        cv::namedWindow("Color Image", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("Depth Image", cv::WINDOW_AUTOSIZE);

        cv::imshow("Color Image", ColorFrame);
        cv::imshow("Depth Image", DepthFrame);

        cv::waitKey(16);
    }
    return 0;
}

int PEACKzed2 :: projectPoints(std::vector<float>& From, std::vector<float>& To)
{
    
    sl::float4 pc_val;
    zpoint_cloud.getValue(From[0], From[1], &pc_val);


    float* ptr = &(pc_val[0]);
    To.assign(ptr, ptr + pc_val.size());
    return 0;
}

int main()
{

    PEACKzed2 test;
    test.init("HD720_SN29530229_18-04-46.svo");
    std::cout << "Device : " << test.get_device_name();
    
    while (test.getFrames() == 0)
    {
        test.showFrames();
    }
    
    
    test.stop();
    return 0;
}