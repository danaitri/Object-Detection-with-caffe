
#include "PyramidDetector.h"
#include "cmath"
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <opencv2/objdetect/objdetect.hpp>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//int counter=0;

void groupRectangles3(std::vector<PyRect>& rectList, int groupThreshold, double eps)
{
	// write output to txt	

	//ofstream w;
	//string output = "/home/danai/data/output.txt";
	//w.open(output.c_str());

	std::vector<int> labels;
	int nclasses = partition(rectList, labels, SimilarPyRects(eps));
	vector<double> dmax(nclasses, 0);

	for (int i = 0; i < labels.size(); i++)
	{	
		
		int cls = labels[i];
		if (rectList[i].weight > dmax[cls])
		dmax[cls] = rectList[i].weight;
	}
	
	//w << "Before Supression RectList size : " << rectList.size() <<  endl;

	for (int i = 0; i < rectList.size(); i++)
	{
		int cls = labels[i];
		if (rectList[i].weight < dmax[cls] * 0.95)
		{
			rectList.erase(rectList.begin() + i);
			labels.erase(labels.begin() + i);
			i--;
		}
	}

	
	std::vector<PyRect> rrects(nclasses);
	std::vector<int> rweights(nclasses, 0);


     //        compute average location (x,y) and (width,height)
     //         for all rectangles that are assigned to the same class / label
     //         i.e. are considered equivalent concerning the SimilarRects()
     //         equivalence predicate (part I)

	int i, j, nlabels = (int)labels.size();
	for (i = 0; i < nlabels; i++)
	{
		int cls = labels[i];
		rrects[cls].x1 += rectList[i].x1;
		rrects[cls].x2 += rectList[i].x2;
		rrects[cls].y1 += rectList[i].y1;
		rrects[cls].y2 += rectList[i].y2;
		//rrects[cls].weight += rectList[i].weight;


        //
        //          rweights[<labelNr>] counts how many detections are assigned to the same
        //         final rectangle
        //         --> could be used as confidence value!
        //

		rweights[cls]++;
	}

	
	for (i = 0; i < nclasses; i++)
	{
		PyRect r = rrects[i];
		double s = 1.0 / rweights[i];
		rrects[i] = r.Scale(s);
		rrects[i].weight = dmax[i];
	}

	rectList.clear();

	for (i = 0; i < nclasses; i++)
	{
		PyRect r1 = rrects[i];
			int n1 = rweights[i];

		// filter out rectangles which don't have enough similar rectangles

		if (n1 < groupThreshold)
			continue;
		// filter out small face rectangles inside large rectangles

		for (j = 0; j < nclasses; j++)
		{
			int n2 = rweights[j];

			if (j == i || n2 < groupThreshold)
				continue;
			PyRect r2 = rrects[j];

			int dx = saturate_cast<int>(r2.width() * eps);
			int dy = saturate_cast<int>(r2.height() * eps);

			if (i != j &&
				r1.x1 >= r2.x1 - dx &&
				r1.y1 >= r2.y1 - dy &&
				r1.x2 <= r2.x2 + dx &&
				r1.y2 <= r2.y2 + dy &&
				(n2 > std::max(3, n1) || n1 < 3))
				break;
	}

		if (j == nclasses)
		{
			rectList.push_back(r1);
		}
	}
}


template <typename Dtype>
PyramidDetector<Dtype>::PyramidDetector(int octave_steps, int scan_size, int stride, int minSize, double threshold, double groupEps, int groupCnt, bool border)
	: _scan_size(scan_size), _steps(octave_steps), _stride(stride), _threshold(threshold), _minSize(minSize), _group_eps(groupEps), _group_count(groupCnt), _useBorder(border)
{
}




template<typename Dtype>
void PyramidDetector<Dtype>::ComputeSizes(Size original)
{
	_sizes.clear();

	int o_size = min(original.height, original.width);
	_increase_ratio = double(_scan_size) / double(_minSize);

	cout << "minSize : " << _minSize << endl;

	double step = 1.0 / ((double)_steps);
	double p = 0;
	double wi = rint(_increase_ratio * original.width);
	double hi = rint(_increase_ratio *  original.height);

	while (true)
	{
		float ratio = (float)pow(2, p);
		int w = rint(ratio * wi);
		int h = rint(ratio * hi);
		int min_dim = std::min<int>(w, h);
		if (min_dim < _scan_size)
			break;
		_sizes.push_back(Size(w, h));
		p -= step;
	}
}


template <typename Dtype>
float PyramidDetector<Dtype>::RatioStep(int step)
{
	float s = 1.0f / ((float)_steps);
	return pow(2.0f, -s*step) * _increase_ratio;
}




template<typename Dtype>
vector<PyRect> PyramidDetector<Dtype>::MultiScaleScan(Mat & img, Net<Dtype>& net)
{
	_sizes.clear();
	int wi = img.cols;
	int hi = img.rows;

	Size imgSize(wi, hi);
	ComputeSizes(imgSize);
	vector<PyRect> rects;
	cout << "Sizes : " << _sizes.size() << endl;
	for (int i = 0; i < _sizes.size() - 1; i++)
	{
		Mat nImage;
		cv::resize(img, nImage, _sizes[i]);
		vector<PyRect> v;
		if (_useBorder)
		{
			int w = _sizes[i].width + _scan_size;
			int h = _sizes[i].height + _scan_size;
			int _border = _scan_size / 2;
			Mat nImage2(h, w, img.type());
			copyMakeBorder(nImage, nImage2, _border, _border, _border, _border, BORDER_CONSTANT, CV_RGB(128, 128, 128));
			v = ScanSingleImage(i, nImage2, net);
			nImage2.release();
		}
		else
		{
			v = ScanSingleImage(i, nImage, net);

		}

		if (v.size() > 0){
                        //cout<<"Group thres = " << _group_count<<endl;
			//PyGroupRectangles(v, _group_count, _group_eps);
                }
		rects.insert(rects.end(), v.begin(), v.end());

		nImage.release();
	}
     
	if (rects.size() > 0) {
            //cout<<"Group thres2 = " << _group_count<<endl;

		PyGroupRectangles(rects, _group_count, _group_eps);
        }

	return rects;
}



template<typename Dtype>
void PyramidDetector<Dtype>::PyGroupRectangles(vector<PyRect>& rects, int groupThreshold, double eps)
{

	groupRectangles3(rects, groupThreshold, eps);

}

template<typename Dtype>
vector<PyRect> PyramidDetector<Dtype>::ScanSingleImage(int step, Mat& img, Net<Dtype>& net)
{
	Blob<Dtype>* b = net.input_blobs()[0];
	vector<int> shape(4);
	shape[0] = 1;
	shape[1] = 3;
	shape[2] = img.rows;
	shape[3] = img.cols;
	b->Reshape(shape);
	MatToBlob(img, b);
	net.Reshape();

	//auto result = net.Forward();
	vector<Blob<Dtype>*> result = net.ForwardPrefilled();
        //cout << result[0]->shape_string() <<", " << result[1]->shape_string() << endl;
	vector<PyRect> rects;
	RectanglesInBlob(result[0], step, rects);


	return rects;
}

template<typename Dtype>
void PyramidDetector<Dtype>::MatToBlob(Mat& img, Blob<Dtype>* blob)
{
	const int img_channels = img.channels();
	const int img_height = img.rows;
	const int img_width = img.cols;

	const int channels = blob->channels();
	const int height = blob->height();
	const int width = blob->width();
	const int num = blob->num();

	CHECK_EQ(channels, img_channels);

	if (height != img_height || width != img_width)
	{
		vector<int> shape(4);
		shape[0] = 1;
		shape[1] = 3;
		shape[2] = img_height;
		shape[3] = img_width;
		blob->Reshape(shape);
	}

	CHECK(img.depth() == CV_8U) << "Image data type must be unsigned byte";

	CHECK_GT(img_channels, 0);

	CHECK(img.data);

	Dtype* transformed_data = blob->mutable_cpu_data();
	int top_index;
	for (int h = 0; h < height; ++h)
	{
		const uchar* ptr = img.ptr<uchar>(h);
		int img_index = 0;
		for (int w = 0; w < width; ++w)
		{
			for (int c = 0; c < img_channels; ++c)
			{
				top_index = (c * height + h) * width + w;
				transformed_data[top_index] = static_cast<Dtype>(ptr[img_index++]);
			}
		}
	}
}

template<typename Dtype>
void PyramidDetector<Dtype>::RectanglesInBlob(Blob<Dtype>* blob, int step, vector<PyRect>& rects)
{
	int wi = blob->width();
	int hi = blob->height();
        
	float ratio = 1.0f / RatioStep(step);
	int size = rint(ratio * _scan_size);
	int _border = _scan_size / 2;
	for (int y = 0; y < hi; y++)
	{
		for (int x = 0; x < wi; x++)
		{
			//int idx = blob->offset(0, 1, y, x);
			Dtype val = blob->data_at(0, 1, y, x);
			if (val > _threshold)
			{
				int xi = lrint((x * _stride - _border)*ratio);
				int yi = lrint((y * _stride - _border)*ratio);
				PyRect new_rect(xi, yi, xi + size, yi + size, val);
				new_rect.level = 1; // step;
				rects.push_back(new_rect);
			}
		}
	}
}


INSTANTIATE_CLASS(PyramidDetector);
