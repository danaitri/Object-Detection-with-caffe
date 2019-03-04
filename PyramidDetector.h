#ifndef PYRAMIDDETECTOR_H
#define PYRAMIDDETECTOR_H

#pragma once

#include "caffe/caffe.hpp"
#include <opencv2/core/core.hpp>
#include <vector>
#include <algorithm>
#include <caffe/net.hpp>



using namespace caffe;
using namespace cv;
using namespace std;



class PyRect{

public:
	int x1;
	int y1;
	int x2;
	int y2;
	int level;
	float weight;
	bool isMatched;

	float Area()
	{
		return float((x2 - x1)*(y2 - y1));
	}


	PyRect(int X1, int Y1, int X2, int Y2, int Level, float W, bool matched) : x1(X1), y1(Y1), x2(X2), y2(Y2), level(Level), weight(W), isMatched(matched)
	{}

	PyRect(int X1, int Y1, int X2, int Y2, float W) : x1(X1), y1(Y1), x2(X2), y2(Y2), weight(W), isMatched(false)
	{}

	PyRect(int X1, int Y1, int X2, int Y2) : x1(X1), y1(Y1), x2(X2), y2(Y2), weight(0.0), isMatched(false)
	{}

	PyRect() :x1(0), y1(0), x2(0), y2(0), weight(0.0), isMatched(false)
	{}

	static PyRect FromCV_Rect(Rect other)
	{
		return PyRect(other.x, other.y, other.x + other.width, other.y + other.height);
	}

	int width() const
	{
		return  x2 - x1;
	}

	int height() const
	{
		return y2 - y1;
	}

	static PyRect Intersection(PyRect A, PyRect B)
	{
		int nx1 = max(A.x1, B.x1);
		int nx2 = min(A.x2, B.x2);
		int ny1 = max(A.y1, B.y1);
		int ny2 = min(A.y2, B.y2);
		if ((nx2 < nx1) || (ny2 < ny1))
		{
			return PyRect(0, 0, 0, 0, 0);
		}
		else
		{
			return PyRect(nx1, ny1, nx2, ny2);
		}
	}

	static float InterSectionRatio(PyRect A, PyRect B)
	{
		PyRect C = PyRect::Intersection(A, B);
		float inSecArea = C.Area();
		float unionArea = A.Area() + B.Area() - C.Area();
		return inSecArea / unionArea;
	}

	Rect ToCV_Rect()
	{
		return Rect(x1, y1, x2 - x1, y2 - y1);
	}

	PyRect Scale(double ratio)
	{
		int nx1 = lround(ratio * x1);
		int nx2 = lround(ratio * x2);
		int ny1 = lround(ratio * y1);
		int ny2 = lround(ratio * y2);
		return PyRect(nx1, ny1, nx2, ny2, level, weight, isMatched);
	}

	PyRect Reduce(double ratio)
	{
		float xc = (x1 + x2) / 2.f;
		float yc = (y1 + y2) / 2.f;
		float nw = width() * ratio;
		float nh = height() * ratio;
		int nx1 = lround(xc - nw / 2);
		int nx2 = lround(xc + nw / 2);
		int ny1 = lround(yc - nh/ 2);
		int ny2 = lround(yc + nh / 2);
		return PyRect(nx1, ny1, nx2, ny2, level, weight, isMatched);
	}

	static int Compare(PyRect a, PyRect b)
	{
		return a.weight > b.weight;
	}
};



class SimilarPyRects
{
public:
	SimilarPyRects(double _eps) : eps(_eps) {}
	inline bool operator() (const PyRect r1, const PyRect r2)
	{
		double delta = eps*(std::min(r1.width(), r2.width()) + std::min(r1.height(), r2.height()) )*0.5;
		return
			std::abs(r1.x1 - r2.x1) <= delta &&
			std::abs(r1.x2 - r2.x2) <= delta &&
			std::abs(r1.y1 - r2.y1) <= delta &&
			std::abs(r1.y2 - r2.y2) <= delta;
	}
	double eps;
};

template <typename Dtype>
class PyramidDetector
{
public:

	PyramidDetector(int octave_steps, int scan_size, int stride, int minSize, double threshold, double groupEps = 0.20, int groupCount = 2, bool border = true);

	void ComputeSizes(Size original);
	void static MatToBlob(Mat& img, Blob<Dtype>* data);
	vector<PyRect> MultiScaleScan(Mat& img, Net<Dtype>& net, Net<Dtype>& net2);
	vector<PyRect> MultiScaleScan(Mat& img, Net<Dtype>& net);
	vector<PyRect> MultiScaleScan2(Mat& img, Net<Dtype>& net);
	vector<PyRect> ScanSingleImage(int step, Mat& img, Net<Dtype>& net);
	vector <PyRect> MakeRectangles(vector<PyRect>& v, Mat &img, Net<Dtype>& net2);
	
	void ChangeMinSize(int minSize)
		{
			_minSize = minSize;
		}



	float RatioStep(int step);
	void RectanglesInBlob(Blob<Dtype>* blob, int step, vector<PyRect>& rects);
        void RectanglesInBlob(Blob<Dtype>* blob, Blob<Dtype>* ratios, int step, vector<PyRect>& rects);

	static void PyGroupRectangles(vector<PyRect>& rects, int groupThreshold, double eps);

private:
	int _scan_size;
	int _steps;
	int _stride;
	int _minSize;
	double _increase_ratio;
	double _threshold;
	double _group_eps;
	int _group_count;
	bool _useBorder;
	vector<Size> _sizes;
};

#endif /* PYRAMIDDETECTOR_H */

