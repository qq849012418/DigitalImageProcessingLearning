#include <opencv.hpp>
#include <sstream>
using namespace std;
using namespace cv;

/*
1、Bmp图像格式简介；
2、把lena 512*512图像灰度级逐级递减8-1显示；
3、计算lena图像的均值方差；
4、把lena图像用近邻、双线性和双三次插值法zoom到2048*2048；
5、把lena和elain图像分别进行水平shear（参数可设置为1.5，或者自行选择）和旋转30度，并采用用近邻、双线性和双三次插值法zoom到2048*2048；

网上找到的方法采用的是opencv2，这里尽量使用opencv3
by Keenster
*/

void Q2(Mat lena)
{
	/*参考https://www.cnblogs.com/asmer-stone/p/5396388.html*/
	Mat temp = lena.clone();
	imshow("原图", temp);
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
		name << "灰度级" << 256/div;
		imshow(name.str(), temp);
	}
}
void Q3(Mat lena)
{
	/*参考https://blog.csdn.net/u012541187/article/details/54571445*/
	Scalar matMean; 
	Scalar matDev; 
	meanStdDev(lena, matMean, matDev);
	float m = matMean.val[0];
	float d = matDev.val[0];
	cout << m << ",\t" << d << endl;
}
void Q4(Mat lena)
{
	/*resize讲解参考https://www.cnblogs.com/korbin/p/5612427.html*/
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
//自制shear函数
void shear(cv::Mat & src, cv::Mat & dst, float dx = 0, float dy = 0) // dx,dy为错切率 
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
	/*错切参考http://answers.opencv.org/question/21262/how-to-do-shear-transformation/*/
	/*旋转参考https://blog.csdn.net/guduruyu/article/details/70799804*/
	Mat sheared;
	Mat rotated;
	shear(lena, sheared, 1.5, 0);
	//Q4(sheared);
	//旋转角度
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
	Q5(original);//改成Q234即可
	waitKey(0);
}
