#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void onTrackbarSlider(int, void*);
Mat mySobel(Mat src_gray);
Mat myLaplacian(Mat src_gray);
Mat myCanny(Mat src_gray);
//�Զ�����ֵ�˲��ο� https://blog.csdn.net/weixin_37720172/article/details/72627543
//��Ÿ�������ֵ
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
	for (int gap = 9 / 2; gap > 0; gap /= 2)//ϣ������
		for (int i = gap; i < 9; ++i)
			for (int j = i - gap; j >= 0 && arr[j] > arr[j + gap]; j -= gap)
				swap(arr[j], arr[j + gap]);
	return arr[4];//������ֵ
}
//��ֵ�˲�����
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
	_dst.copyTo(dst);//����
}
void Q1(int ksize)
{
	Mat src, dst1, dst2;
	src = imread("test1.pgm");
	imshow("test1ԭͼ", src);
	medianBlur(src, dst1, ksize);
	imshow("�Դ���ֵ�˲�", dst1);
	MedianFlitering(src, dst2);
	imshow("�Զ�����ֵ�˲�", dst2);
}

//�Զ����˹�˲��ο� https://blog.csdn.net/weixin_37720172/article/details/72843238
double **getGuassionArray(int size, double sigma) {
	int i, j;
	double sum = 0.0;
	int center = size; //�Ե�һ���������Ϊԭ�㣬������ĵ������

	double **arr = new double*[size];//����һ��size*size��С�Ķ�ά����
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
			//��Ե�����д���
			if ((i - 1) > 0 && (i + 1) < _src.rows && (j - 1) > 0 && (j + 1) < _src.cols) {
				arr = getGuassionArray(ksize, sigma);//�Զ���õ���Ȩֵ����
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
	imshow("test1ԭͼ", src);
	GaussianBlur(src, dst1, Size(ksize, ksize), sigma);
	imshow("�Դ���˹�˲�", dst1);
	myGaussian(src, dst2, ksize, sigma);
	imshow("�Զ����˹�˲�", dst2);
	
	
}
//�����ڱκ͸������˲����� https://blog.csdn.net/u013108511/article/details/69214856 �����޸�
Mat Highboost(Mat src) {  /*ƽ������*/
	Mat in = src;
	Mat out;
	out.create(in.size(),in.type());
	GaussianBlur(in, out, Size(3, 3), 1);
	float c = 2;//�˴�����1Ϊ�����ڱΣ�����1Ϊ�������˲���С��1Ϊ��ǿ������ģ��Ĺ���
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

//ȫ�ֱ�����������
Mat g_grayImage, g_dstImage;
//��¼�������ĵ�ǰλ��
int g_slider_pos = 0;
//�տ�ʼʱ�������޵�ֵ
int m_lowerLimit = 1;
int m_upperLimit = 4;
//�������ص�����
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

	

	//����Ч��ͼ
	imshow("Ч��", g_dstImage);
}
//Sobel�ο� https://www.cnblogs.com/nipan/p/4141409.html
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
	//ʹ��Laplace����
	Laplacian(src_gray, dst, CV_16S, 3, 1, 0, BORDER_DEFAULT);

	//�������ֵ���������ת����8λ
	convertScaleAbs(dst, abs_dst);
	return abs_dst;
}
Mat myCanny(Mat src_gray)
{
	Mat edge;
	//blur(src_gray, edge, Size(3, 3));
	Canny(src_gray, edge, 50, 150, 3);
	////��5����g_dstImage�ڵ�����Ԫ������Ϊ0 
	//dst = Scalar::all(0);

	////��6��ʹ��Canny��������ı�Եͼg_cannyDetectedEdges��Ϊ���룬����ԭͼg_srcImage����Ŀ��ͼg_dstImage��
	//src_gray.copyTo(dst, edge);
	//return dst;
	return edge;

}
void Q3(void)
{
	//��Ե����ܲο� https://blog.csdn.net/poem_qianmo/article/details/25560901
	g_grayImage = imread("test4 copy.bmp");
	namedWindow("Ч��", WINDOW_AUTOSIZE);
	//����������
	createTrackbar(
		"type",
		"Ч��",
		&g_slider_pos,
		m_upperLimit,
		onTrackbarSlider
	);	
	//��ʼ���Զ���� ��ֵ�ص�����
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