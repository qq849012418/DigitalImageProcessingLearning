#include <opencv.hpp>
#include <sstream>
using namespace std;
using namespace cv;

/*
1��Bmpͼ���ʽ��飻
2����lena 512*512ͼ��Ҷȼ��𼶵ݼ�8-1��ʾ��
3������lenaͼ��ľ�ֵ���
4����lenaͼ���ý��ڡ�˫���Ժ�˫���β�ֵ��zoom��2048*2048��
5����lena��elainͼ��ֱ����ˮƽshear������������Ϊ1.5����������ѡ�񣩺���ת30�ȣ��������ý��ڡ�˫���Ժ�˫���β�ֵ��zoom��2048*2048��

�����ҵ��ķ������õ���opencv2�����ﾡ��ʹ��opencv3
by Keenster
*/

void Q2(Mat lena)
{
	/*�ο�https://www.cnblogs.com/asmer-stone/p/5396388.html*/
	Mat temp = lena.clone();
	imshow("ԭͼ", temp);
	float div;
	for (int i = 1; i < 8; i++)
	{
		div = pow(2, i);
		for (int y = 0; y < temp.rows; y++)
		{
			uchar* data = temp.ptr<uchar>(y);
			for (int x = 0; x < temp.cols*temp.channels(); x++)
			{
				data[x] = int(data[x] / div) * div;
			}
		}
		stringstream name;
		name << "�Ҷȼ�" << 256/div;
		imshow(name.str(), temp);
	}
}
void Q3(Mat lena)
{
	/*�ο�https://blog.csdn.net/u012541187/article/details/54571445*/
	Scalar matMean; 
	Scalar matDev; 
	meanStdDev(lena, matMean, matDev);
	float m = matMean.val[0];
	float d = matDev.val[0];
	cout << m << ",\t" << d << endl;
}
void Q4(Mat lena)
{
	/*resize����ο�https://www.cnblogs.com/korbin/p/5612427.html*/
	Mat a, b, c;
	resize(lena, a, Size(2048, 2048), 0, 0, INTER_NEAREST);
	resize(lena, b, Size(2048, 2048), 0, 0, INTER_LINEAR);
	resize(lena, c, Size(2048, 2048), 0, 0, INTER_CUBIC);
	namedWindow("lena_NN", CV_WINDOW_NORMAL);
	namedWindow("lena_LINEAR", CV_WINDOW_NORMAL);
	namedWindow("lena_CUBIC", CV_WINDOW_NORMAL);
	imshow("lena_NN", a);
	imshow("lena_LINEAR", b);
	imshow("lena_CUBIC", c);

}
//����shear����
void shear(cv::Mat & src, cv::Mat & dst, float dx = 0, float dy = 0) // dx,dyΪ������ 
{
	const int rows = src.rows; // rows == H (Y--->)
	const int cols = src.cols; // cols == W (X--->)
	int maxXOffset = abs(cols * dx);
	int maxYOffset = abs(rows * dy);
	dst = Mat::ones(rows+ maxYOffset, cols+ maxXOffset, src.type());
	Vec3b *p;
	for (int Y = 0; Y < dst.rows; ++Y)
	{
		p = dst.ptr<Vec3b>(Y);
		for (int X = 0; X < dst.cols; ++X)
		{
			int X0 = X + dx * Y - maxXOffset;
			int Y0 = Y + dy * X - maxYOffset;
			if (X0 >= 0 && Y0 >= 0 && X0 < cols && Y0 < rows)
			{
				p[X] = src.ptr<Vec3b>(Y0)[X0];
			}
		}
	}
}

void Q5(Mat lena)
{
	/*���вο�http://answers.opencv.org/question/21262/how-to-do-shear-transformation/*/
	/*��ת�ο�https://blog.csdn.net/guduruyu/article/details/70799804*/
	Mat sheared;
	Mat rotated;
	shear(lena, sheared, 1.5, 0);
	//Q4(sheared);
	//��ת�Ƕ�
	double angle = 30;
	Size src_sz = sheared.size();
	Size dst_sz(src_sz.width, src_sz.height);
	Point2f center(src_sz.width / 2, src_sz.height / 2);
	Mat rot_mat = getRotationMatrix2D(center, angle, 1.0);
	warpAffine(sheared, rotated, rot_mat, dst_sz);
	Q4(rotated);
}
void main()
{
	Mat original;
	original = imread("lena.bmp");
	Q5(original);//�ĳ�Q234����
	waitKey(0);
}
