#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <sstream>

using namespace cv;
using namespace std;

void showHist(Mat image,String name)
{
	//定义变量
	MatND dstHist;
	int dims = 1;
	float hrange[] = { 0,255 };
	const float * ranges[] = { hrange };
	int size = 256;
	int channels = 0;
	//参考https ://blog.csdn.net/huayunhualuo/article/details/81868014 
	calcHist(&image, 1, &channels, Mat(), dstHist, dims, &size, ranges);
	int scale = 1;
	Mat dstImage(size*scale, size, CV_8U, Scalar(0));
	//获取最大值和最小值
	double minValue = 0;
	double maxValue = 0;
	minMaxLoc(dstHist, &minValue, &maxValue, 0, 0);
	//绘制出直方图
	int hpt = saturate_cast<int>(0.9*size);
	for (int i = 0; i < 256; i++)
	{
		float binValue = dstHist.at<float>(i);
		int realValue = saturate_cast<int>(binValue*hpt / maxValue);
		rectangle(dstImage, Point(i*scale, size - 1), Point((i + 1)*scale - 1, size - realValue), Scalar(255));

	}
	imshow(name, dstImage);
}
void Q1(Mat* images)
{
	for (int i = 0; i < 8; i++)
	{
		stringstream name;
		name << "处理图" << i;
		showHist(images[i], name.str());
	}
}

void Q2(Mat* images)
{
	for (int i = 0; i < 8; i++)
	{
		stringstream name;
		name << "处理图" << i;
		equalizeHist(images[i], images[i]);
		imshow(name.str(), images[i]);
		
		//showHist(images[i], name.str());
	}
}
//直方图匹配代码1，比较底层
bool histMatch_Value(Mat matSrc, Mat matDst, Mat &matRet)
{
	//参考https://www.cnblogs.com/konglongdanfo/p/9215091.html
	if (matSrc.empty() || matDst.empty() || 1 != matSrc.channels() || 1 != matDst.channels())
		return false;
	int nHeight = matDst.rows;
	int nWidth = matDst.cols;
	int nDstPixNum = nHeight * nWidth;
	int nSrcPixNum = 0;

	int arraySrcNum[256] = { 0 };                // 源图像各灰度统计个数
	int arrayDstNum[256] = { 0 };                // 目标图像个灰度统计个数
	double arraySrcProbability[256] = { 0.0 };   // 源图像各个灰度概率
	double arrayDstProbability[256] = { 0.0 };   // 目标图像各个灰度概率
												 // 统计源图像
	for (int j = 0; j < nHeight; j++)
	{
		for (int i = 0; i < nWidth; i++)
		{
			arrayDstNum[matDst.at<uchar>(j, i)]++;
		}
	}
	// 统计目标图像
	nHeight = matSrc.rows;
	nWidth = matSrc.cols;
	nSrcPixNum = nHeight * nWidth;
	for (int j = 0; j < nHeight; j++)
	{
		for (int i = 0; i < nWidth; i++)
		{
			arraySrcNum[matSrc.at<uchar>(j, i)]++;
		}
	}
	// 计算概率
	for (int i = 0; i < 256; i++)
	{
		arraySrcProbability[i] = (double)(1.0 * arraySrcNum[i] / nSrcPixNum);
		arrayDstProbability[i] = (double)(1.0 * arrayDstNum[i] / nDstPixNum);
	}
	// 构建直方图均衡映射
	int L = 256;
	int arraySrcMap[256] = { 0 };
	int arrayDstMap[256] = { 0 };
	for (int i = 0; i < L; i++)
	{
		double dSrcTemp = 0.0;
		double dDstTemp = 0.0;
		for (int j = 0; j <= i; j++)
		{
			dSrcTemp += arraySrcProbability[j];
			dDstTemp += arrayDstProbability[j];
		}
		arraySrcMap[i] = (int)((L - 1) * dSrcTemp + 0.5);// 减去1，然后四舍五入
		arrayDstMap[i] = (int)((L - 1) * dDstTemp + 0.5);// 减去1，然后四舍五入
	}
	// 构建直方图匹配灰度映射
	int grayMatchMap[256] = { 0 };
	for (int i = 0; i < L; i++) // i表示源图像灰度值
	{
		int nValue = 0;    // 记录映射后的灰度值
		int nValue_1 = 0;  // 记录如果没有找到相应的灰度值时，最接近的灰度值
		int k = 0;
		int nTemp = arraySrcMap[i];
		for (int j = 0; j < L; j++) // j表示目标图像灰度值
		{
			// 因为在离散情况下，之风图均衡化函数已经不是严格单调的了，
			// 所以反函数可能出现一对多的情况，所以这里做个平均。
			if (nTemp == arrayDstMap[j])
			{
				nValue += j;
				k++;
			}
			if (nTemp < arrayDstMap[j])
			{
				nValue_1 = j;
				break;
			}
		}
		if (k == 0)// 离散情况下，反函数可能有些值找不到相对应的，这里去最接近的一个值
		{
			nValue = nValue_1;
			k = 1;
		}
		grayMatchMap[i] = nValue / k;
	}
	// 构建新图像
	matRet = Mat::zeros(nHeight, nWidth, CV_8UC1);
	for (int j = 0; j < nHeight; j++)
	{
		for (int i = 0; i < nWidth; i++)
		{
			matRet.at<uchar>(j, i) = grayMatchMap[matSrc.at<uchar>(j, i)];
		}
	}
	return true;
}
//自行编写的匹配代码2（未完成）
void HistMatch(Mat img_s, Mat img_m, Mat& img_o)
{

	MatND srcHist;
	MatND baseHist;
	int dims = 1;
	float hrange[] = { 0,255 };
	const float * ranges[] = { hrange };
	int size = 256;
	int channels = 0;
	calcHist(&img_s, 1, &channels, Mat(), srcHist, dims, &size, ranges);
	calcHist(&img_m, 1, &channels, Mat(), baseHist, dims, &size, ranges);
	normalize(baseHist, baseHist, 0, 1, NORM_MINMAX, -1, Mat());
	normalize(srcHist, srcHist, 0, 1, NORM_MINMAX, -1, Mat());

}
void Q3(Mat* images)
{
	imshow("citywall", images[8]);
	for (int i = 0; i < 2; i++)
	{
		Mat result;
		stringstream name;
		name << "citywall" << i;
		histMatch_Value(images[i], images[8], result);
		imshow(name.str(), result);
	}
}
void Q4(Mat* images)
{
	//不知道做的对不对，网上无原题
	//参考1、https://www.2cto.com/kf/201612/577046.html
	//2、https://blog.csdn.net/zxc024000/article/details/51252073
	//关于滤波器内核 https://blog.csdn.net/Bigat/article/details/80792865
	Mat elain = images[9].clone();
	imshow("elain原图像", elain);
	Mat lena = images[10].clone();
	imshow("lena原图像", lena);
	Mat imageEnhance,result,result2;
	//Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, 0, 5, 0, 0, -1, 0);
	Laplacian(elain, result, CV_32F, 7);
	result.convertTo(result2, CV_8U, 1, -50);
	imageEnhance = elain - result2;
	imshow("elain增强", imageEnhance);
	Laplacian(lena, result, CV_32F, 7);
	result.convertTo(result2, CV_8U, 1, -50);
	imageEnhance = lena - result2;
	imshow("lena增强", imageEnhance);
}
//关于threshold（）函数 https://blog.csdn.net/qq_35294564/article/details/81325136
//全局变量声明部分
Mat g_grayImage, g_dstImage;
//记录滚动条的当前位置
int g_slider_pos_low = 0;
int g_slider_pos_high = 255;
//刚开始时的上下限的值
int m_lowerLimit = 0;
int m_upperLimit = 255;
//滚动条回调函数
void onTrackbarSlider_low(int, void*)
{
	//调用阈值函数
	threshold(g_grayImage, g_dstImage, g_slider_pos_low, 255, THRESH_TOZERO);

	//更新效果图
	imshow("效果", g_dstImage);
}

void onTrackbarSlider_high(int, void*)
{
	//调用阈值函数
	threshold(g_grayImage, g_dstImage, g_slider_pos_high, 255, THRESH_TOZERO_INV);

	//更新效果图
	imshow("效果", g_dstImage);
}

void Q5(Mat* images)
{
	g_grayImage = images[9].clone();
	namedWindow("效果", WINDOW_AUTOSIZE);
	//创建滚动条
	createTrackbar(
		"m_lowerLimit",
		"效果",
		&g_slider_pos_low,
		255,
		onTrackbarSlider_low
	);
	createTrackbar(
		"m_upperLimit",
		"效果",
		&g_slider_pos_high,
		255,
		onTrackbarSlider_high
	);
	//初始化自定义的 阈值回调函数
	onTrackbarSlider_low(0, 0);
	onTrackbarSlider_high(0, 0);
	
}
int main()
{
	Mat* images = new Mat[11];
	images[8] = imread("citywall.bmp",0);
	images[0] = imread("citywall1.bmp", 0);
	images[1] = imread("citywall2.bmp", 0);
	images[9] = imread("elain.bmp", 0);
	images[2] = imread("elain1.bmp", 0);
	images[3] = imread("elain2.bmp", 0);
	images[4] = imread("elain3.bmp", 0);
	images[10] = imread("lena.bmp", 0);
	images[5] = imread("lena1.bmp", 0);
	images[6] = imread("lena2.bmp", 0);
	images[7] = imread("lena4.bmp", 0);
	Q5(images);
	waitKey(0);
	return 0;
}