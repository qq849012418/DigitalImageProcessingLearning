#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void onTrackbarSlider(int, void*);
Mat mySobel(Mat src_gray);
Mat myLaplacian(Mat src_gray);
Mat myCanny(Mat src_gray);
//自定义中值滤波参考 https://blog.csdn.net/weixin_37720172/article/details/72627543
//求九个数的中值
uchar Median(uchar n1, uchar n2, uchar n3, uchar n4, uchar n5,
	uchar n6, uchar n7, uchar n8, uchar n9) {
	uchar arr[9];
	arr[0] = n1;
	arr[1] = n2;
	arr[2] = n3;
	arr[3] = n4;
	arr[4] = n5;
	arr[5] = n6;
	arr[6] = n7;
	arr[7] = n8;
	arr[8] = n9;
	for (int gap = 9 / 2; gap > 0; gap /= 2)//希尔排序
		for (int i = gap; i < 9; ++i)
			for (int j = i - gap; j >= 0 && arr[j] > arr[j + gap]; j -= gap)
				swap(arr[j], arr[j + gap]);
	return arr[4];//返回中值
}
//中值滤波函数
void MedianFlitering(const Mat &src, Mat &dst) {
	if (!src.data)return;
	Mat _dst(src.size(), src.type());
	for (int i = 0; i<src.rows; ++i)
		for (int j = 0; j < src.cols; ++j) {
			if ((i - 1) > 0 && (i + 1) < src.rows && (j - 1) > 0 && (j + 1) < src.cols) {
				_dst.at<Vec3b>(i, j)[0] = Median(src.at<Vec3b>(i, j)[0], src.at<Vec3b>(i + 1, j + 1)[0],
					src.at<Vec3b>(i + 1, j)[0], src.at<Vec3b>(i, j + 1)[0], src.at<Vec3b>(i + 1, j - 1)[0],
					src.at<Vec3b>(i - 1, j + 1)[0], src.at<Vec3b>(i - 1, j)[0], src.at<Vec3b>(i, j - 1)[0],
					src.at<Vec3b>(i - 1, j - 1)[0]);
				_dst.at<Vec3b>(i, j)[1] = Median(src.at<Vec3b>(i, j)[1], src.at<Vec3b>(i + 1, j + 1)[1],
					src.at<Vec3b>(i + 1, j)[1], src.at<Vec3b>(i, j + 1)[1], src.at<Vec3b>(i + 1, j - 1)[1],
					src.at<Vec3b>(i - 1, j + 1)[1], src.at<Vec3b>(i - 1, j)[1], src.at<Vec3b>(i, j - 1)[1],
					src.at<Vec3b>(i - 1, j - 1)[1]);
				_dst.at<Vec3b>(i, j)[2] = Median(src.at<Vec3b>(i, j)[2], src.at<Vec3b>(i + 1, j + 1)[2],
					src.at<Vec3b>(i + 1, j)[2], src.at<Vec3b>(i, j + 1)[2], src.at<Vec3b>(i + 1, j - 1)[2],
					src.at<Vec3b>(i - 1, j + 1)[2], src.at<Vec3b>(i - 1, j)[2], src.at<Vec3b>(i, j - 1)[2],
					src.at<Vec3b>(i - 1, j - 1)[2]);
			}
			else
				_dst.at<Vec3b>(i, j) = src.at<Vec3b>(i, j);
		}
	_dst.copyTo(dst);//拷贝
}
void Q1(int ksize)
{
	Mat src, dst1, dst2;
	src = imread("test1.pgm");
	imshow("test1原图", src);
	medianBlur(src, dst1, ksize);
	imshow("自带中值滤波", dst1);
	MedianFlitering(src, dst2);
	imshow("自定义中值滤波", dst2);
}

//自定义高斯滤波参考 https://blog.csdn.net/weixin_37720172/article/details/72843238
double **getGuassionArray(int size, double sigma) {
	int i, j;
	double sum = 0.0;
	int center = size; //以第一个点的坐标为原点，求出中心点的坐标

	double **arr = new double*[size];//建立一个size*size大小的二维数组
	for (i = 0; i < size; ++i)
		arr[i] = new double[size];

	for (i = 0; i < size; ++i)
		for (j = 0; j < size; ++j) {
			arr[i][j] = exp(-((i - center)*(i - center) + (j - center)*(j - center)) / (sigma*sigma * 2));
			sum += arr[i][j];
		}
	for (i = 0; i < size; ++i)
		for (j = 0; j < size; ++j)
			arr[i][j] /= sum;
	return arr;
}
void myGaussian(const Mat _src, Mat &_dst, int ksize, double sigma) {
	if (!_src.data) return;
	double **arr;
	Mat tmp(_src.size(), _src.type());
	for (int i = 0; i < _src.rows; ++i)
		for (int j = 0; j < _src.cols; ++j) {
			//边缘不进行处理
			if ((i - 1) > 0 && (i + 1) < _src.rows && (j - 1) > 0 && (j + 1) < _src.cols) {
				arr = getGuassionArray(ksize, sigma);//自定义得到的权值数组
				tmp.at<Vec3b>(i, j)[0] = 0;
				tmp.at<Vec3b>(i, j)[1] = 0;
				tmp.at<Vec3b>(i, j)[2] = 0;
				for (int x = 0; x < 3; ++x) {
					for (int y = 0; y < 3; ++y) {
						tmp.at<Vec3b>(i, j)[0] += arr[x][y] * _src.at<Vec3b>(i + 1 - x, j + 1 - y)[0];
						tmp.at<Vec3b>(i, j)[1] += arr[x][y] * _src.at<Vec3b>(i + 1 - x, j + 1 - y)[1];
						tmp.at<Vec3b>(i, j)[2] += arr[x][y] * _src.at<Vec3b>(i + 1 - x, j + 1 - y)[2];
					}
				}
			}
		}
	tmp.copyTo(_dst);
}

void Q2(int ksize, double sigma)
{
	Mat src, dst1, dst2;
	src = imread("test2.tif");
	imshow("test1原图", src);
	GaussianBlur(src, dst1, Size(ksize, ksize), sigma);
	imshow("自带高斯滤波", dst1);
	myGaussian(src, dst2, ksize, sigma);
	imshow("自定义高斯滤波", dst2);
	
	
}
//非锐化掩蔽和高提升滤波参照 https://blog.csdn.net/u013108511/article/details/69214856 进行修改
Mat Highboost(Mat src) {  /*平滑处理*/
	Mat in = src;
	Mat out;
	out.create(in.size(),in.type());
	GaussianBlur(in, out, Size(3, 3), 1);
	float c = 2;//此处等于1为非锐化掩蔽，大于1为高提升滤波，小于1为不强调非锐化模板的贡献
	for (int i = 0; i<out.rows; i++) {
		for (int j = 0; j<out.cols; j++) {
			for (int k = 0; k < 3; k++)
			{
				Scalar _f_x_y = out.at<Vec3b>(i, j)[k];
				Scalar f_x_y = in.at<Vec3b>(i, j)[k];
				int d = f_x_y.val[0] + c * (f_x_y.val[0] - _f_x_y.val[0]);
				if (d>255) {
					d = 255;
				}
				else if (d<0) {
					d = 0;
				}

				out.at<Vec3b>(i, j)[k] = d;
			}
			
		}
	}
	return out;
}

//全局变量声明部分
Mat g_grayImage, g_dstImage;
//记录滚动条的当前位置
int g_slider_pos = 0;
//刚开始时的上下限的值
int m_lowerLimit = 1;
int m_upperLimit = 4;
//滚动条回调函数
//type:0=original,1:unsharp masking, 2:Sobel edge detector,3:Laplace edge detection,4:Canny algorithm.
void onTrackbarSlider(int, void*)
{
	switch (g_slider_pos)
	{
	case 0:
		g_dstImage = g_grayImage.clone();
		break;
	case 1:
		g_dstImage = Highboost(g_grayImage);
		break;
	case 2:
		g_dstImage = mySobel(g_grayImage);
		break;
	case 3:
		g_dstImage = myLaplacian(g_grayImage);
		break;
	case 4:
		g_dstImage = myCanny(g_grayImage);
		break;

	}

	

	//更新效果图
	imshow("效果", g_dstImage);
}
//Sobel参考 https://www.cnblogs.com/nipan/p/4141409.html
// Generate grad_x and grad_y
   Mat grad_x, grad_y;
   Mat grad;
   Mat abs_grad_x, abs_grad_y;
   int scale = 1;
   int delta = 0;
   int ddepth = CV_16S;
Mat mySobel(Mat src_gray)
{
	GaussianBlur(src_gray, src_gray, Size(3, 3), 0, 0, BORDER_DEFAULT);
	/// Gradient X
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	/// Gradient Y
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	/// Total Gradient (approximate)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
	return grad;
}
Mat myLaplacian(Mat src_gray)
{
	Mat dst;
	Mat abs_dst;
	GaussianBlur(src_gray, src_gray, Size(3, 3), 0, 0, BORDER_DEFAULT);
	//使用Laplace函数
	Laplacian(src_gray, dst, CV_16S, 3, 1, 0, BORDER_DEFAULT);

	//计算绝对值，并将结果转换成8位
	convertScaleAbs(dst, abs_dst);
	return abs_dst;
}
Mat myCanny(Mat src_gray)
{
	Mat edge;
	//blur(src_gray, edge, Size(3, 3));
	Canny(src_gray, edge, 50, 150, 3);
	////【5】将g_dstImage内的所有元素设置为0 
	//dst = Scalar::all(0);

	////【6】使用Canny算子输出的边缘图g_cannyDetectedEdges作为掩码，来将原图g_srcImage拷到目标图g_dstImage中
	//src_gray.copyTo(dst, edge);
	//return dst;
	return edge;

}
void Q3(void)
{
	//边缘检测总参考 https://blog.csdn.net/poem_qianmo/article/details/25560901
	g_grayImage = imread("test4 copy.bmp");
	namedWindow("效果", WINDOW_AUTOSIZE);
	//创建滚动条
	createTrackbar(
		"type",
		"效果",
		&g_slider_pos,
		m_upperLimit,
		onTrackbarSlider
	);	
	//初始化自定义的 阈值回调函数
	onTrackbarSlider(0, 0);

	
}
int main()
{
	//Q1(3);
	//Q2(3,1.5);
	Q3();
	waitKey(0);
	return 0;
}