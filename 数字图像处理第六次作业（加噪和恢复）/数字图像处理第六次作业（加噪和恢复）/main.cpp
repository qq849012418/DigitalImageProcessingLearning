//利用程序给原图像增加椒盐噪声和高斯噪声
//参考 https://blog.csdn.net/qq_34784753/article/details/69379135 进行修改

#include <cstdlib>
#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace cv;
using namespace std;

Mat addSaltNoise(const Mat srcImage, int n);
double generateGaussianNoise(double m, double sigma);
Mat addGaussianNoise(Mat &srcImag);
void SaltNoiseTest(Mat srcImage)
{
	Mat dstImage = addSaltNoise(srcImage, 3000);//修改数字可控制密度
	imshow("添加椒盐噪声的图像", dstImage);
	//存储图像
	imwrite("salt_pepper_Image.jpg", dstImage);
	medianBlur(dstImage, dstImage, 5);
	imshow("去除椒盐噪声的图像", dstImage);
}
void GaussianNoiseTest(Mat srcImage)
{
	Mat dstImage = addGaussianNoise(srcImage);
	imshow("添加高斯噪声后的图像", dstImage);
	GaussianBlur(dstImage, dstImage, Size(5, 5), 1);
	imshow("去除高斯噪声后的图像", dstImage);
}
int main()
{
	Mat srcImage = imread("lena.bmp");
	if (!srcImage.data)
	{
		cout << "读入图像有误！" << endl;
		system("pause");
		return -1;
	}
	imshow("原图像", srcImage);
	//GaussianNoiseTest(srcImage);
	SaltNoiseTest(srcImage);
	waitKey();
	return 0;
}

Mat addSaltNoise(const Mat srcImage, int n)
{
	Mat dstImage = srcImage.clone();
	for (int k = 0; k < n; k++)
	{
		//随机取值行列
		int i = rand() % dstImage.rows;
		int j = rand() % dstImage.cols;
		//图像通道判定
		if (dstImage.channels() == 1)
		{
			dstImage.at<uchar>(i, j) = 255;		//盐噪声
		}
		else
		{
			dstImage.at<Vec3b>(i, j)[0] = 255;
			dstImage.at<Vec3b>(i, j)[1] = 255;
			dstImage.at<Vec3b>(i, j)[2] = 255;
		}
	}
	for (int k = 0; k < n; k++)
	{
		//随机取值行列
		int i = rand() % dstImage.rows;
		int j = rand() % dstImage.cols;
		//图像通道判定
		if (dstImage.channels() == 1)
		{
			dstImage.at<uchar>(i, j) = 0;		//椒噪声
		}
		else
		{
			dstImage.at<Vec3b>(i, j)[0] = 0;
			dstImage.at<Vec3b>(i, j)[1] = 0;
			dstImage.at<Vec3b>(i, j)[2] = 0;
		}
	}
	return dstImage;
}
//生成高斯噪声
//mu高斯函数的偏移，sigma高斯函数的标准差
double generateGaussianNoise(double mu, double sigma)
{
	//定义小值
	const double epsilon = numeric_limits<double>::min();
	static double z0, z1;
	static bool flag = false;
	flag = !flag;
	//flag为假构造高斯随机变量X
	if (!flag)
		return z1 * sigma + mu;
	double u1, u2;
	//构造随机变量
	do
	{
		u1 = rand() * (1.0 / RAND_MAX);
		u2 = rand() * (1.0 / RAND_MAX);
	} while (u1 <= epsilon);
	//flag为真构造高斯随机变量
	z0 = sqrt(-2.0*log(u1))*cos(2 * CV_PI*u2);
	z1 = sqrt(-2.0*log(u1))*sin(2 * CV_PI*u2);
	return z0 * sigma + mu;
}

//为图像添加高斯噪声
Mat addGaussianNoise(Mat &srcImag)
{
	Mat dstImage = srcImag.clone();
	int channels = dstImage.channels();
	int rowsNumber = dstImage.rows;
	int colsNumber = dstImage.cols*channels;
	//判断图像的连续性
	if (dstImage.isContinuous())
	{
		colsNumber *= rowsNumber;
		rowsNumber = 1;
	}
	for (int i = 0; i < rowsNumber; i++)
	{
		for (int j = 0; j < colsNumber; j++)
		{
			//添加高斯噪声
			int val = dstImage.ptr<uchar>(i)[j] +
				generateGaussianNoise(2, 0.8) * 32;
			if (val < 0)
				val = 0;
			if (val>255)
				val = 255;
			dstImage.ptr<uchar>(i)[j] = (uchar)val;
		}
	}
	return dstImage;
}
