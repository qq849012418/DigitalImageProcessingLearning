#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <sstream>

using namespace cv;
using namespace std;

void showHist(Mat image,String name)
{
	//�������
	MatND dstHist;
	int dims = 1;
	float hrange[] = { 0,255 };
	const float * ranges[] = { hrange };
	int size = 256;
	int channels = 0;
	//�ο�https ://blog.csdn.net/huayunhualuo/article/details/81868014 
	calcHist(&image, 1, &channels, Mat(), dstHist, dims, &size, ranges);
	int scale = 1;
	Mat dstImage(size*scale, size, CV_8U, Scalar(0));
	//��ȡ���ֵ����Сֵ
	double minValue = 0;
	double maxValue = 0;
	minMaxLoc(dstHist, &minValue, &maxValue, 0, 0);
	//���Ƴ�ֱ��ͼ
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
		name << "����ͼ" << i;
		showHist(images[i], name.str());
	}
}

void Q2(Mat* images)
{
	for (int i = 0; i < 8; i++)
	{
		stringstream name;
		name << "����ͼ" << i;
		equalizeHist(images[i], images[i]);
		imshow(name.str(), images[i]);
		
		//showHist(images[i], name.str());
	}
}
//ֱ��ͼƥ�����1���Ƚϵײ�
bool histMatch_Value(Mat matSrc, Mat matDst, Mat &matRet)
{
	//�ο�https://www.cnblogs.com/konglongdanfo/p/9215091.html
	if (matSrc.empty() || matDst.empty() || 1 != matSrc.channels() || 1 != matDst.channels())
		return false;
	int nHeight = matDst.rows;
	int nWidth = matDst.cols;
	int nDstPixNum = nHeight * nWidth;
	int nSrcPixNum = 0;

	int arraySrcNum[256] = { 0 };                // Դͼ����Ҷ�ͳ�Ƹ���
	int arrayDstNum[256] = { 0 };                // Ŀ��ͼ����Ҷ�ͳ�Ƹ���
	double arraySrcProbability[256] = { 0.0 };   // Դͼ������Ҷȸ���
	double arrayDstProbability[256] = { 0.0 };   // Ŀ��ͼ������Ҷȸ���
												 // ͳ��Դͼ��
	for (int j = 0; j < nHeight; j++)
	{
		for (int i = 0; i < nWidth; i++)
		{
			arrayDstNum[matDst.at<uchar>(j, i)]++;
		}
	}
	// ͳ��Ŀ��ͼ��
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
	// �������
	for (int i = 0; i < 256; i++)
	{
		arraySrcProbability[i] = (double)(1.0 * arraySrcNum[i] / nSrcPixNum);
		arrayDstProbability[i] = (double)(1.0 * arrayDstNum[i] / nDstPixNum);
	}
	// ����ֱ��ͼ����ӳ��
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
		arraySrcMap[i] = (int)((L - 1) * dSrcTemp + 0.5);// ��ȥ1��Ȼ����������
		arrayDstMap[i] = (int)((L - 1) * dDstTemp + 0.5);// ��ȥ1��Ȼ����������
	}
	// ����ֱ��ͼƥ��Ҷ�ӳ��
	int grayMatchMap[256] = { 0 };
	for (int i = 0; i < L; i++) // i��ʾԴͼ��Ҷ�ֵ
	{
		int nValue = 0;    // ��¼ӳ���ĻҶ�ֵ
		int nValue_1 = 0;  // ��¼���û���ҵ���Ӧ�ĻҶ�ֵʱ����ӽ��ĻҶ�ֵ
		int k = 0;
		int nTemp = arraySrcMap[i];
		for (int j = 0; j < L; j++) // j��ʾĿ��ͼ��Ҷ�ֵ
		{
			// ��Ϊ����ɢ����£�֮��ͼ���⻯�����Ѿ������ϸ񵥵����ˣ�
			// ���Է��������ܳ���һ�Զ�������������������ƽ����
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
		if (k == 0)// ��ɢ����£�������������Щֵ�Ҳ������Ӧ�ģ�����ȥ��ӽ���һ��ֵ
		{
			nValue = nValue_1;
			k = 1;
		}
		grayMatchMap[i] = nValue / k;
	}
	// ������ͼ��
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
//���б�д��ƥ�����2��δ��ɣ�
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
	//��֪�����ĶԲ��ԣ�������ԭ��
	//�ο�1��https://www.2cto.com/kf/201612/577046.html
	//2��https://blog.csdn.net/zxc024000/article/details/51252073
	//�����˲����ں� https://blog.csdn.net/Bigat/article/details/80792865
	Mat elain = images[9].clone();
	imshow("elainԭͼ��", elain);
	Mat lena = images[10].clone();
	imshow("lenaԭͼ��", lena);
	Mat imageEnhance,result,result2;
	//Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, 0, 5, 0, 0, -1, 0);
	Laplacian(elain, result, CV_32F, 7);
	result.convertTo(result2, CV_8U, 1, -50);
	imageEnhance = elain - result2;
	imshow("elain��ǿ", imageEnhance);
	Laplacian(lena, result, CV_32F, 7);
	result.convertTo(result2, CV_8U, 1, -50);
	imageEnhance = lena - result2;
	imshow("lena��ǿ", imageEnhance);
}
//����threshold�������� https://blog.csdn.net/qq_35294564/article/details/81325136
//ȫ�ֱ�����������
Mat g_grayImage, g_dstImage;
//��¼�������ĵ�ǰλ��
int g_slider_pos_low = 0;
int g_slider_pos_high = 255;
//�տ�ʼʱ�������޵�ֵ
int m_lowerLimit = 0;
int m_upperLimit = 255;
//�������ص�����
void onTrackbarSlider_low(int, void*)
{
	//������ֵ����
	threshold(g_grayImage, g_dstImage, g_slider_pos_low, 255, THRESH_TOZERO);

	//����Ч��ͼ
	imshow("Ч��", g_dstImage);
}

void onTrackbarSlider_high(int, void*)
{
	//������ֵ����
	threshold(g_grayImage, g_dstImage, g_slider_pos_high, 255, THRESH_TOZERO_INV);

	//����Ч��ͼ
	imshow("Ч��", g_dstImage);
}

void Q5(Mat* images)
{
	g_grayImage = images[9].clone();
	namedWindow("Ч��", WINDOW_AUTOSIZE);
	//����������
	createTrackbar(
		"m_lowerLimit",
		"Ч��",
		&g_slider_pos_low,
		255,
		onTrackbarSlider_low
	);
	createTrackbar(
		"m_upperLimit",
		"Ч��",
		&g_slider_pos_high,
		255,
		onTrackbarSlider_high
	);
	//��ʼ���Զ���� ��ֵ�ص�����
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