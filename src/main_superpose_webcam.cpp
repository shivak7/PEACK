#include "PEACKTracker_CubeMos.h"
#include "PEACK_Camera_generic.h"
#include <iostream>
#include <sys/stat.h>
#include <UL_Track.h>

#define PI 3.14159265

const std::map<std::string, unsigned int> BODY_PARTS_LUT{
	{ "Nose", 0 },
	{ "Neck", 1 },
	{ "RShoulder",2 },
	{ "RElbow", 3 },
	{ "RWrist", 4 },
	{ "LShoulder", 5 },
	{ "LElbow", 6 },
	{ "LWrist", 7 },
	{ "RHip", 8 },
	{ "RKnee", 9 },
	{ "RAnkle", 10 },
	{ "LHip", 11 },
	{ "LKnee", 12 },
	{ "LAnkle", 13 },
	{ "REye", 14 },
	{ "LEye", 15 },
	{ "REar", 16 },
	{ "LEar", 17 }
};

inline bool file_exists(const std::string& name) {
	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);
}

float dist(std::vector<float> &x, std::vector<float> &y)
{
	return sqrt(powf((x[0] - y[0]), 2) + powf((x[1] - y[1]), 2));
}

float dot(std::vector<float> &x, std::vector<float> &y)
{
	return (x[0] * y[0]) + (x[1] * y[1]);
}

float norm(std::vector<float> &x)
{
	return sqrt(powf(x[0], 2) + powf(x[1], 2));
}

std::vector<float> vec(std::vector<float> &x, std::vector<float> &y)
{
	std::vector<float> ret;

	ret.push_back(x[0] - y[0]);
	ret.push_back(x[1] - y[1]);

	return ret;
}

std::pair<float, float> pair_center(std::vector<float> &x, std::vector<float> &y)
{
	return std::pair<float, float>((x[0] + y[0]) / 2.0, (x[1] + y[1]) / 2.0);
}



int draw_face(PEACKTracker& T)
{

	std::vector<float> Reye, Leye, Nose, Center, pCenter;
	T.getPartKeyPoint(14, Reye);
	T.getPartKeyPoint(15, Leye);
	T.getPartKeyPoint(0, Nose);

	std::pair<float,float> p = pair_center(Reye, Leye);
	Center.push_back(p.first);
	Center.push_back(p.second);
	
	pCenter.push_back(Nose[0]);
	pCenter.push_back(Nose[1] + 50);

	std::vector<float> vec1 = vec(Center, Nose);
	std::vector<float> vec2 = vec(pCenter, Nose);

	float angle = acosf(dot(vec1, vec2) / (norm(vec1) * norm(vec2))) * 180.0 / PI;
	if (Center[0] > Nose[0])
		angle = angle * -1;

	std::cout << "Angle: " << angle << std::endl;
	cv::Mat disp = T.dev->ColorFrame.clone();
	MyEllipse(disp, Center[0], Center[1], 1.5 * dist(Reye, Leye), 4 * dist(Center, Nose), angle);
	cv::Mat res;

	cv::addWeighted(disp, 0.4, T.dev->ColorFrame, 0.6, 1, res);

	cv::imshow("Render", res);
	int ch = cv::waitKey(1);

	if (ch == 27)
		T.dev->CurrentStatus = PEACK_DEVICE_STATUS_STOPPED;
}

int draw_UL_Target_pose1(PEACKTracker& T)
{

	std::vector<float> LShldr, RShldr, RElb, RWrist, Center1, Center2, pseud1, pseud2;

	T.getPartKeyPoint(2, RShldr);
	T.getPartKeyPoint(3, RElb);
	T.getPartKeyPoint(4, RWrist);
	T.getPartKeyPoint(5, LShldr);

	std::pair<float, float> p = pair_center(RShldr, RElb);
	Center1.push_back(p.first);
	Center1.push_back(p.second);

	pseud1.push_back(RShldr[0] + 50);
	pseud1.push_back(RShldr[1]);

	p = pair_center(RElb, RWrist);
	Center2.push_back(p.first);
	Center2.push_back(p.second);

	pseud2.push_back(RElb[0] + 50);
	pseud2.push_back(RElb[1]);


	float limb_width = 0.4 * dist(RShldr, LShldr);
	cv::Mat disp = T.dev->ColorFrame.clone();


}

int draw_UL_pose(PEACKTracker& T)
{
	std::vector<float> LShldr, RShldr, RElb, RWrist, Center1, Center2, pseud1, pseud2;

	T.getPartKeyPoint(2, RShldr);
	T.getPartKeyPoint(3, RElb);
	T.getPartKeyPoint(4, RWrist);
	T.getPartKeyPoint(5, LShldr);

	std::pair<float, float> p = pair_center(RShldr, RElb);
	Center1.push_back(p.first);
	Center1.push_back(p.second);

	pseud1.push_back(RShldr[0] + 50);
	pseud1.push_back(RShldr[1]);

	p = pair_center(RElb, RWrist);
	Center2.push_back(p.first);
	Center2.push_back(p.second);

	pseud2.push_back(RElb[0] + 50);
	pseud2.push_back(RElb[1]);

	
	float limb_width = 0.4*dist(RShldr, LShldr);
	cv::Mat disp = T.dev->ColorFrame.clone();

	std::vector<float> vec1 = vec(RElb, RShldr);
	std::vector<float> vec2 = vec(pseud1, RShldr);
	float angle1 = acosf(dot(vec1, vec2) / (norm(vec1) * norm(vec2))) * 180.0 / PI;
	if (RElb[1] < RShldr[1])
		angle1 = angle1 * -1;
	cv::RotatedRect rRect1 = cv::RotatedRect(cv::Point2f(Center1[0], Center1[1]), cv::Size2f(dist(RShldr, RElb), limb_width), angle1);
	cv::Point2f vertices2f[4];
	rRect1.points(vertices2f);
	std::vector<cv::Point> vertices;
	for (int i = 0; i < 4; ++i) {
		vertices.push_back(vertices2f[i]);
	}
	cv::fillConvexPoly(disp, vertices, cv::Scalar(0, 255, 255), 8);

	vec1 = vec(RWrist,RElb);
	vec2 = vec(pseud2, RElb);
	float angle2 = acosf(dot(vec1, vec2) / (norm(vec1) * norm(vec2))) * 180.0 / PI;
	if (RWrist[1] < RElb[1])
		angle2 = angle2 * -1;
	cv::RotatedRect rRect2 = cv::RotatedRect(cv::Point2f(Center2[0], Center2[1]), cv::Size2f(dist(RWrist, RElb), limb_width), angle2);
	rRect2.points(vertices2f);
	vertices.clear();
	for (int i = 0; i < 4; ++i) {
		vertices.push_back(vertices2f[i]);
	}
	cv::fillConvexPoly(disp, vertices, cv::Scalar(0, 255, 255), 8);

	cv::Mat res;

	//cv::addWeighted(disp, 0.4, T.dev->ColorFrame, 0.6, 1, res);

	cv::imshow("Render", T.dev->ColorFrame);
	int ch = cv::waitKey(1);

	if (ch == 27)
		T.dev->CurrentStatus = PEACK_DEVICE_STATUS_STOPPED;

}

int main(int argc, char** argv)
{

	
		std::string OutFile = "Out.csv";			
			
		PEACKDevice* cam = new PEACKGenCam;
		cam->init(1920, 1080, 60);

		PEACKTracker * Tracker1 = new PEACKTracker_CubeMos;
		Tracker1->init(PEACKTracker_Pose, PEACKDetection_Mode_2D, cam, OutFile);

		ULTracker ULT(Tracker1, true);
		//for (int i = 0; i < 100; i++)
		auto start = std::chrono::high_resolution_clock::now();
		cv::namedWindow("Render", cv::WINDOW_NORMAL);
		cv::setWindowProperty("Render", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
		cv::Mat res;
		while (Tracker1->dev->CurrentStatus != PEACK_DEVICE_STATUS_STOPPED)
		{
			Tracker1->getFrameFromDevice();
			//Tracker1.dev->showFrames();

			Tracker1->predictKeypoints();
			Tracker1->processKeypoints2D();
			
			ULT.Update();
			res = ULT.DrawBBoxes(Tracker1->dev->ColorFrame, ULT.ActualBBox);
			

			cv::imshow("Render", res);
			int ch = cv::waitKey(1);

			if (ch == 27)
				Tracker1->dev->CurrentStatus = PEACK_DEVICE_STATUS_STOPPED;

			//draw_face(Tracker1);
			//draw_UL_pose(*Tracker1);
			//std::cout << "Device run status: " << Tracker1.dev->CurrentStatus << std::endl;
		}

		auto end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double> elapsed = end - start;
		std::cout << "Execution time: " << elapsed.count() << std::endl;
		
		Tracker1->release();
		//delete Tracker1;
	//system("pause");
	return 0;
}