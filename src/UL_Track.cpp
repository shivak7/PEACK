#include "UL_Track.h"

ULTracker::ULTracker(PEACKTracker* tracker, bool UseFilt)
{
	T = tracker;
	UseFilter = UseFilt;
	
	if (UseFilt)
	{
		for (int i = 0; i < NUM_UPPERLIMBS * 3; i++)
		{
			Filters[i].Init(TRACK_FILTER_ORD);
			//std::cout << "Initializing filter: "<< i << std::endl;
		}
	}
}

int ULTracker::Init(PEACKTracker* tracker, bool UseFilt)
{
	T = tracker;
	UseFilter = UseFilt;
	if (UseFilt)
	{
		for (int i = 0; i < NUM_UPPERLIMBS * 3; i++)
		{
			Filters[i].Init(TRACK_FILTER_ORD);
		}
	}
}

int ULTracker::UpdateFilters()
{
	int i = 0;
	int k = 0;
	for (i = 0; i < 3; i++)
		Filters[i+k].Insert(Pre.RShldr[i]);
	k += i;
	for (i = 0; i < 3; i++)
		Filters[i + k].Insert(Pre.LShldr[i]);
	k += i;
	for (i = 0; i < 3; i++)
		Filters[i + k].Insert(Pre.RElb[i]);
	k += i;
	for (i = 0; i < 3; i++)
		Filters[i + k].Insert(Pre.LElb[i]);
	k += i;
	for (i = 0; i < 3; i++)
		Filters[i + k].Insert(Pre.RWrist[i]);
	k += i;
	for (i = 0; i < 3; i++)
		Filters[i + k].Insert(Pre.LWrist[i]);
	
	

	Current.RShldr.clear(); Current.RElb.clear(); Current.RWrist.clear();
	Current.LShldr.clear(); Current.LElb.clear(); Current.LWrist.clear();

	i = k = 0;
	for (i = 0; i < 3; i++)
		Current.RShldr.push_back(Filters[i + k].Median());
	k += i;
	for (i = 0; i < 3; i++)
		Current.LShldr.push_back(Filters[i + k].Median());
	k += i;
	for (i = 0; i < 3; i++)
		Current.RElb.push_back(Filters[i + k].Median());
	k += i;
	for (i = 0; i < 3; i++)
		Current.LElb.push_back(Filters[i + k].Median());
	k += i;
	for (i = 0; i < 3; i++)
		Current.RWrist.push_back(Filters[i + k].Median());
	k += i;
	for (i = 0; i < 3; i++)
		Current.LWrist.push_back(Filters[i + k].Median());
	
	
	return 0;
}

int ULTracker::Update()
{
	Pre.RShldr.clear(); Pre.RElb.clear(); Pre.RWrist.clear();
	Pre.LShldr.clear(); Pre.LElb.clear(); Pre.LWrist.clear();

	T->getPartKeyPoint(T->getMappingFromString("RShoulder"), Pre.RShldr);
	T->getPartKeyPoint(T->getMappingFromString("LShoulder"), Pre.LShldr);
	T->getPartKeyPoint(T->getMappingFromString("RElbow"), Pre.RElb);
	T->getPartKeyPoint(T->getMappingFromString("LElbow"), Pre.LElb);
	T->getPartKeyPoint(T->getMappingFromString("RWrist"), Pre.RWrist);
	T->getPartKeyPoint(T->getMappingFromString("LWrist"), Pre.LWrist);

	std::cout << "Step 1" << std::endl;
	if (UseFilter)
	{
		UpdateFilters();
		calculateBBoxes(Current, ActualBBox);
	}
	else
		calculateBBoxes(Pre, ActualBBox);
	std::cout << "Step 2" << std::endl;
	return 0;
}

cv::RotatedRect getBBox(std::vector<float>& joint1, std::vector<float>& joint2, float box_width, bool right)
{
	std::vector<float> Center1, Center2, pseud1, pseud2;

	std::pair<float, float> p = PMath::pair_center(joint1, joint2);
	Center1.push_back(p.first);
	Center1.push_back(p.second);

	if(right)
		pseud1.push_back(joint1[0] + 50);
	else
		pseud1.push_back(joint1[0] - 50);

	pseud1.push_back(joint1[1]);

	std::vector<float> vec1 = PMath::vec(joint2, joint1);
	std::vector<float> vec2 = PMath::vec(pseud1, joint1);
	float angle1 = acosf(PMath::dot(vec1, vec2) / (PMath::norm(vec1) * PMath::norm(vec2))) * 180.0 / PMath::PI();

	if (joint2[1] < joint1[1])
		angle1 = angle1 * -1;

	if(!right)
		angle1 = angle1 * -1;

	cv::RotatedRect rRect1 = cv::RotatedRect(cv::Point2f(Center1[0], Center1[1]), cv::Size2f(PMath::dist(joint1, joint2), box_width), angle1);

	return rRect1;
}

int ULTracker::calculateBBoxes(ULStruct& UL, BBox& BoundingBox)
{

	float limb_width = 0.4 * PMath::dist(UL.RShldr, UL.LShldr);
	BoundingBox.RUpperArm = getBBox(UL.RShldr, UL.RElb, limb_width, true); // Right Upper limb
	BoundingBox.RForeArm = getBBox(UL.RElb, UL.RWrist, limb_width, true); // Right fore limb
	BoundingBox.LUpperArm = getBBox(UL.LShldr, UL.LElb, limb_width, false); // Right Upper limb
	BoundingBox.LForeArm = getBBox(UL.LElb, UL.LWrist, limb_width, false); // Right fore limb
	

	return 0;
}

int drawBBox(cv::Mat& img, cv::RotatedRect&rect1)
{
	cv::Point2f vertices2f[4];
	rect1.points(vertices2f);
	std::vector<cv::Point> vertices;
	for (int i = 0; i < 4; ++i) {
		vertices.push_back(vertices2f[i]);
	}
	cv::fillConvexPoly(img, vertices, cv::Scalar(0, 255, 255), 8);

	return 0;
}

void MyEllipse(cv::Mat& img, int cx, int cy, int w, int h, double angle)
{
	int thickness = 2;
	int lineType = 8;
	cv::ellipse(img,
		cv::Point(cx, cy),
		cv::Size(w, h),
		angle,
		0,
		360,
		cv::Scalar(0, 255, 255, 90),
		-1,
		lineType);
}

cv::Mat ULTracker::DrawBBoxes(cv::Mat& img, BBox& box)
{
	cv::Mat res;
	cv::Mat disp = img.clone();

	drawBBox(disp, box.RUpperArm);
	drawBBox(disp, box.RForeArm);

	drawBBox(disp, box.LUpperArm);
	drawBBox(disp, box.LForeArm);

	DrawJointsHands(disp, Current);

	cv::addWeighted(disp, 0.4, img, 0.6, 1, res);
	return res;
}

int ULTracker::DrawJointsHands(cv::Mat& img, ULStruct& UL)
{
	
	float limb_width = 0.4 * PMath::dist(UL.RShldr, UL.LShldr);
	//Left elbow
	int cx = UL.LElb[0];
	int cy = UL.LElb[1];
	MyEllipse(img, cx, cy, limb_width/2.0, limb_width/2.0, 0);
	
	//Right elbow
	cx = UL.RElb[0];
	cy = UL.RElb[1];
	MyEllipse(img, cx, cy, limb_width / 2.0, limb_width / 2.0, 0);

	//Left hand
	float hand_rad = limb_width / 2;
	std::vector<float> v1 = PMath::vec(UL.LWrist, UL.LElb);
	float mag = PMath::norm(v1);
	cx = UL.LWrist[0] + (v1[0] / mag) * (limb_width / 1.5);
	cy = UL.LWrist[1] + (v1[1] / mag) * (limb_width / 1.5);
	MyEllipse(img, cx, cy, hand_rad, hand_rad, 0);

	//Right hand
	v1 = PMath::vec(UL.RWrist, UL.RElb);
	mag = PMath::norm(v1);
	cx = UL.RWrist[0] + (v1[0] / mag) * hand_rad;
	cy = UL.RWrist[1] + (v1[1] / mag) * hand_rad;
	MyEllipse(img, cx, cy, hand_rad, hand_rad, 0);

	return 0;
}