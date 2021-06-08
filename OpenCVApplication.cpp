// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <queue>
using namespace std;


void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

Mat_<uchar> openImg()
{
	char fname[MAX_PATH];
	Mat_<uchar> src;
	openFileDlg(fname);
	src = imread(fname);
	//	imshow("image", src);
	return src;

}


void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}
////////////////////////////////////  LAB 1 /////////////////////////////////////////
void testNegativeImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = 255 - val;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testGreyPlus()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = val + 30;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("grey image", dst);
		waitKey();
	}
}

void testGreyMultiplication()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = val * 2;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("grey image", dst);
		waitKey();
	}
}

void createCologImg()
{

	Mat img(256, 256, CV_8UC3);
	int height = img.rows;
	int width = img.cols;


	for (int i = 0; i < height / 2; i++)
	{
		for (int j = 0; j < width / 2; j++)
		{

			img.at<Vec3b>(i, j)[0] = 255;
			img.at<Vec3b>(i, j)[1] = 255;
			img.at<Vec3b>(i, j)[2] = 255;
		}
	}
	for (int i = 0; i < height / 2; i++)
	{
		for (int j = width / 2; j < width; j++)
		{
			img.at<Vec3b>(i, j)[0] = 0;
			img.at<Vec3b>(i, j)[1] = 0;
			img.at<Vec3b>(i, j)[2] = 255;
		}
	}
	for (int i = height / 2; i < height; i++)
	{
		for (int j = width / 2; j < width; j++)
		{
			img.at<Vec3b>(i, j)[0] = 0;
			img.at<Vec3b>(i, j)[1] = 255;
			img.at<Vec3b>(i, j)[2] = 255;
		}
	}
	for (int i = height / 2; i < height; i++)
	{
		for (int j = 0; j < width / 2; j++)
		{
			img.at<Vec3b>(i, j)[0] = 0;
			img.at<Vec3b>(i, j)[1] = 204;
			img.at<Vec3b>(i, j)[2] = 0;
		}
	}


	imshow("color image", img);
	waitKey();

}

void matriceInversa() {
	//matrice initiala
	float vals[9] = { 1,-1,1,2,0,3,1,1,-2 };
	Mat M(3, 3, CV_32FC1, vals);
	std::cout << M << std::endl;

	//calc determinat matrice
	float determinant = vals[0] * vals[4] * vals[8] + vals[1] * vals[5] * vals[6] + vals[3] * vals[7] * vals[2]
		- vals[6] * vals[4] * vals[2] - vals[7] * vals[5] * vals[0] - vals[3] * vals[1] * vals[8];

	//calculez mat transpusa
	float tr[9];
	Mat N(3, 3, CV_32FC1, tr);
	for (int i = 0; i < 9; i++)
	{
		int m = i / 3;
		int n = i % 3;
		tr[i] = vals[3 * n + m];
	}
	std::cout << "Transpusa\n";
	std::cout << N << std::endl;

	//calculez mat adiacenta
	float d11 = (tr[4] * tr[8] - tr[5] * tr[7]);
	float d12 = (-1) * (tr[1] * tr[8] - tr[2] * tr[7]);
	float d13 = (tr[1] * tr[5] - tr[4] * tr[2]);

	float d21 = (-1) * (tr[3] * tr[8] - tr[5] * tr[6]);
	float d22 = (tr[0] * tr[8] - tr[6] * tr[2]);
	float d23 = (-1) * (tr[0] * tr[5] - tr[2] * tr[3]);

	float d31 = (tr[3] * tr[7] - tr[4] * tr[6]);
	float d32 = (-1) * (tr[7] * tr[0] - tr[6] * tr[1]);
	float d33 = (tr[0] * tr[4] - tr[1] * tr[3]);

	float adj[9] = { d11, d21,d31, d12, d22, d32, d13, d23, d33 };
	Mat A(3, 3, CV_32FC1, adj);
	std::cout << "Matricea adiacenta\n";
	std::cout << A << std::endl;

	//calculez si afisez matricea inversa
	float inv[9] = { d11 / determinant, d21 / determinant,d31 / determinant, d12 / determinant, d22 / determinant, d32 / determinant, d13 / determinant, d23 / determinant, d33 / determinant };
	Mat I(3, 3, CV_32FC1, inv);
	std::cout << "Matricea inversa\n";
	std::cout << I << std::endl;


	system("pause");
	waitKey();


}
////////////////////////////////////////////////////////////////////////////////////////////////////
void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar* lpSrc = src.data;
		uchar* lpDst = dst.data;
		int w = (int)src.step; // no dword alignment is done !!!
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i * w + j];
				lpDst[i * w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}


void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}








void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int hi = i * width * 3 + j * 3;
				int gi = i * width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, dst, gauss;
		src = imread(fname, IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int)k * pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);
		imshow("input image", src);
		imshow("canny", dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}

	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame, edges, 40, 100, 3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];

	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;

		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115) { //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess)
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
	}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

///////// LABORATOR 2 /////////
void copyRGBcolors()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat conv1 = Mat(height, width, CV_8UC3);
		Mat conv2 = Mat(height, width, CV_8UC3);
		Mat conv3 = Mat(height, width, CV_8UC3);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b blue, green, red;
				Vec3b v3 = src.at<Vec3b>(i, j);
				blue[0] = v3[0];
				blue[1] = 0;
				blue[2] = 0;

				green[1] = v3[1];
				green[0] = 0;
				green[2] = 0;

				red[2] = v3[2];
				red[0] = 0;
				red[1] = 0;

				conv1.at<Vec3b>(i, j) = blue;
				conv2.at<Vec3b>(i, j) = green;
				conv3.at<Vec3b>(i, j) = red;
			}
		}
		imshow("input image", src);
		imshow("image1", conv1);
		imshow("image2", conv2);
		imshow("image3", conv3);
		waitKey();
	}
}


void rgbToGrey()
{

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat conv1 = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				conv1.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}
		imshow("input image", src);
		imshow("image1", conv1);
		waitKey(0);
	}
}


void greyScaleToBinary()
{
	char fname[MAX_PATH];
	int th;
	printf("Dati threshold:  ");
	scanf("%d", &th);
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				if (src.at<uchar>(i, j) < th) dst.at<uchar>(i, j) = 0;
				else dst.at<uchar>(i, j) = 255;
			}

		imshow("input image", src);
		imshow("binaryimage", dst);
		waitKey(0);
	}
}

void rgbToHSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		Mat H_mat = Mat(height, width, CV_8UC1);
		Mat S_mat = Mat(height, width, CV_8UC1);
		Mat V_mat = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				Vec3b v3 = src.at<Vec3b>(i, j);
				float B, G, R;
				B = v3[0];
				G = v3[1];
				R = v3[2];
				float r, g, b;
				r = (float)R / 255;
				g = (float)G / 255;
				b = (float)B / 255;

				float minim, maxim;
				float max1, min1;

				max1 = max(g, b);
				maxim = max(r, max1);

				min1 = min(g, b);
				minim = min(r, min1);

				float C = maxim - minim;
				//value:
				float V = maxim;
				float H; float S;
				//saturation:
				if (V != 0)
					S = C / V;
				else
				 //negru
					S = 0;
				//hue
				if (C != 0) {
					if (maxim == r) H = 60 * (g - b) / C;
					if (maxim == g) H = 120 + 60 * (b - r) / C;
					if (maxim == b) H = 240 + 60 * (r - g) / C;
				}
				else
					H = 0;
				if (H < 0)
					H = H + 360;

				float H_norm = H * 255 / 360;
				float S_norm = S * 255;
				float V_norm = V * 255;

				H_mat.at<uchar>(i, j) = (uchar)H_norm;
				S_mat.at<uchar>(i, j) = (uchar)S_norm;
				V_mat.at<uchar>(i, j) = (uchar)V_norm;
			}
		}


		imshow("input image", src);
		imshow("H", H_mat);
		imshow("S", S_mat);
		imshow("V", V_mat);

		waitKey();
	}
}

int isInside(int a, int b)
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;
		printf("Lungime-Latime img: %d %d\n", height, width);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (a >= 0 && a <= height && b >= 0 && b <= width)	return 0;
				else return 1;
			}
		}
	}
	system("pause");
}
//////////////////////////////////////

///////////////////LABORATOR 3 /////////////////


int calcHist()
{
	int hist[256];
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;
		for (int i = 0; i < 255; i++) {
			hist[i] = 0;
		}
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				hist[src.at<uchar>(i, j)]++;
			}
		}
		for (int i = 0; i < 255; i++) {
			printf("%d  ", hist[i]);
		}
		showHistogram("MyHist", hist, 255, 200);
	}
	return 0;
}

int normHist()
{
	int hist[256];
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;
		for (int i = 0; i < 255; i++) {
			hist[i] = 0;
		}
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				hist[src.at<uchar>(i, j)]++;

			}
		}
		int M = height * width;
		for (int i = 0; i < 255; i++) {
			hist[i] = (float)(hist[i] / M);
			
		}
		for (int i = 0; i < 255; i++) {
			printf("%f  ", hist[i]);
		}
		showHistogram("MyHist", hist, 255, 200);
	}
	return 0;
}




int find_closest_histogram_maximum(int oldpixel)
{
	return oldpixel;
}

void floydSteinberg() {
	int hist[256];
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;
		for (int i = 0; i < height - 1; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int oldpixel = src.at<uchar>(i, j);
				int newpixel = find_closest_histogram_maximum(oldpixel);
				src.at<uchar>(i, j) = newpixel;
				int error = oldpixel - newpixel;
				if (isInside(i, j + 1))
					src.at<uchar>(i, j + 1) = src.at<uchar>(i, j + 1) + 7 * error / 16;
				if (isInside(i + 1, j - 1))
					src.at<uchar>(i + 1, j - 1) = src.at<uchar>(i + 1, j - 1) + 3 * error / 16;
				if (isInside(i + 1, j))
					src.at<uchar>(i + 1, j) = src.at<uchar>(i + 1, j) + 5 * error / 16;
				if (isInside(i + 1, j + 1))
					src.at<uchar>(i + 1, j + 1) = src.at<uchar>(i, j + 1) + error / 16;
			}
		}
	}
}
////////////////////////////////////
void praguriMultiple()
{
	int hist[256];
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;
		for (int i = 0; i < height - 1; i++)
		{
			for (int j = 0; j < width; j++)
			{
				hist[src.at<uchar>(i, j)]++;
			}
		}
		int M = height * width;
		for (int i = 0; i < 255; i++) {
			hist[i] = (float)(hist[i] / M);
		}
		int WH = 5;
		int fereastra = 2 * WH + 1;
		int TH = 0.0003;
		int maxime[256];
		maxime[0] = 1;
		maxime[255] = 1;
		for (int i = 0 + WH; i < 255 - WH; i++)
		{
			//media
			int v; int k;
		}
	}
}



void centru_masa()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		int i, j;
		int area = 0;

		//arie
		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++)
			{
				if (src.at<uchar>(i, j) == 0)
				{
					area++;
				}
			}
		}
		printf("Aria =%d\n", area);


		//centrul de masa
		float r_aux = 0.0;
		float c_aux = 0.0;
		float r;
		float c;
		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++)
			{
				if (src.at<uchar>(i, j) == 0)
				{
					r_aux += i;
					c_aux += j;

				}
			}
		}
		r = r_aux / area;
		c = c_aux / area;

		printf("r=%f\n", r);
		printf("c=%f\n", c);

		//axa de alungire
		float sum1 = 0;
		float sum2 = 0;
		float sum3 = 0;
		float num = 0;
		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++)
			{
				if (src.at<uchar>(i, j) == 0)
				{
					sum1 = sum1 + (i - r) * (j - c);
					sum2 = sum2 + (j - c) * (j - c);
					sum3 = sum3 + (i - r) * (i - r);
				}
			}
		}
		sum1 = sum1 * 2;
		num = sum2 - sum3;
		double phi = atan2(sum1, num) / 2;
		printf("Unghiul phi =%f\n", phi);

		//perimetru
		float perimetrul = 0;
		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++)
			{
				if (src.at<uchar>(i, j) == 0 && (src.at<uchar>(i + 1, j) != 0 || src.at<uchar>(i, j - 1) != 0 || src.at<uchar>(i, j + 1) != 0 || src.at<uchar>(i - 1, j) != 0))
				{
					perimetrul++;


				}
			}
		}
		perimetrul = perimetrul * (PI / 4);
		printf("Permietrul= %f\n", perimetrul);

		//factor de subtiere al obiectului
		float t;
		t = 4 * PI * ((float)area / (perimetrul * perimetrul));
		printf("Factor de subtiere=%f\n", t);


		//elongatia
		float Cmin = -10000;
		float Cmax = 99999;
		float Rmin = -10000;
		float Rmax = 99999;

		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++)
			{
				if (Cmin > j) Cmin = j;
				if (Cmax < j) Cmax = j;
				if (Rmin > i) Rmin = i;
				if (Rmax < i) Rmax = i;
			}
		}
		float R = (Cmax - Cmin + 1) / (Rmax - Rmin + 1);
		printf("elongatia =%f\n", R);

		//proiectie orizontala
		float pr_oriz = 0;
		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++) {
				if (src.at<uchar>(i, 0) == 0)
					pr_oriz++;
			}
		}
		printf("proiectie orizontala= %f\n", pr_oriz);

		//proiectie verticala
		float pr_vert = 0;
		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++) {
				if (src.at<uchar>(0, j) == 0)
					pr_vert++;
			}
		}
		printf("proiectie verticala= %f\n", pr_vert);
		printf("-----------------");

		imshow("input image", src);
		waitKey();
	}
}

void laborator4() {

	int aria = 0, perimetru = 0;
	double numarator = 0, numitor = 0, numitor1 = 0, numitor2 = 0;
	float sum1 = 0, sum2 = 0;
	double factor_subtiere, factor_aspect;
	double ri, ci;

	Mat_ <uchar> img = imread("C:/Users/Rebeca/Desktop/PI/Lab4_img/triunghi_vfsus.bmp", 1);
	Mat_ <Vec3b> copie = Mat(img.rows, img.cols, CV_8UC3);
	copie.setTo(Vec3b(255, 255, 255));

	//aria + colorat verde

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {

			if (img(i, j) == 0)
			{
				aria++;
				copie(i, j)[0] = 0;
				copie(i, j)[1] = 204;
				copie(i, j)[2] = 0;
			}

		}
	}

	//centru de masa

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {

			if (img(i, j) == 0)
			{
				sum1 += i;
				sum2 += j;

			}

			ri = sum1 / aria;
			ci = sum2 / aria;
		}
	}

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {

			if (i == (int)ri && j == (int)ci)
			{
				for (int a = 1; a <= 4; a++) {

					copie(i + a, j)[0] = 0;
					copie(i + a, j)[1] = 0;
					copie(i + a, j + a)[2] = 0;

					copie(i - a, j)[0] = 0;
					copie(i - a, j)[1] = 0;
					copie(i - a, j)[2] = 0;

					copie(i, j + a)[0] = 0;
					copie(i, j + a)[1] = 0;
					copie(i, j + a)[2] = 0;

					copie(i, j - a)[0] = 0;
					copie(i, j - a)[1] = 0;
					copie(i, j - a)[2] = 0;

				}
			}
		}
	}


	// axa de alungire


	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {

			if (img(i, j) == 0)
			{
				numarator += (i - ri) * (j - ci);
				numitor1 += (j - ci) * (j - ci);
				numitor2 += (i - ri) * (i - ri);
			}
		}
	}

	numarator = 2 * numarator;
	numitor = numitor1 - numitor2;

	double unghi = atan2(numarator, numitor) / 2;

	line(copie, Point(ci + 30 * cos(unghi), ri + 30 * sin(unghi)), Point(ci - 30 * cos(unghi), ri - 30 * sin(unghi)), Scalar(0, 0, 255));

	//perimetru

	for (int i = 1; i < img.rows; i++) {
		for (int j = 1; j < img.cols; j++) {

			if (img(i, j) == 0 && (img(i + 1, j) != 0 || img(i, j + 1) != 0 || img(i, j - 1) != 0 || img(i - 1, j) != 0))
			{
				perimetru++;

				copie(i, j)[0] = 0;
				copie(i, j)[1] = 0;
				copie(i, j)[1] = 0;
			}
		}
	}

	perimetru = perimetru * (PI / 4);


	// Factorul de subţiere 
	factor_subtiere = 4 * PI * ((double)aria / (perimetru * perimetru));

	cout << "aria= " << aria << endl;
	cout << "rc= " << ri << endl;
	cout << "cc= " << ci << endl;
	cout << "perimetru= " << perimetru << endl;
	cout << "factor subtiere= " << factor_subtiere << endl;

	imshow("Aria", copie);
	waitKey(0);
}

// laborator  5

void traversare_latime() {

	
	Mat_<uchar> img = imread("C:/Users/Rebeca/Desktop/PI/LAB5/letters.bmp", 1);
	Mat_<int> labels(img.rows, img.cols);
	int label = 0;
	labels.setTo(0);

	Mat_<Vec3b> imgRez = Mat::zeros(img.rows, img.cols, CV_8UC3);

	int di[4] = { 0, -1, 0, 1 };
	int dj[4] = { -1, 0, 1, 0 };


	for (int i = 0; i < img.rows - 1; i++) {
		for (int j = 0; j < img.cols - 1; j++) {
			if (img(i, j) == 0 && labels(i, j) == 0) {
				label++;
				queue<Point2i> Q;
				labels(i, j) = label;
				Q.push({ i,j });
				while (!Q.empty()) {
					Point2i q = Q.front();
					Q.pop();
					for (int k = 0; k < 4; k++) {
						Point2i neighbors = { q.x + di[k],q.y + dj[k] };
						if (neighbors.x >= 0 && neighbors.y >= 0 && neighbors.x < img.rows && neighbors.y < img.cols) {
							if (img(q.x + di[k], q.y + dj[k]) == 0 && labels(q.x + di[k], q.y + dj[k]) == 0) {
								labels(q.x + di[k], q.y + dj[k]) = label;
								Q.push(neighbors);

							}
						}
					}
				}

			}
		}
	}

	//culoare
	vector<Vec3b> culori = vector<Vec3b>(label + 1);
	for (int i = 0; i <= label; i++) {
		uchar r = rand() % 255;
		uchar g = rand() % 255;
		uchar b = rand() % 255;
		culori.at(i) = Vec3b(b, g, r);
	}

	for (int i = 0; i < imgRez.rows; i++) {
		for (int j = 0; j < imgRez.cols; j++) {

			if (labels.at<int>(i, j) > 0) {
				imgRez.at<Vec3b>(i, j) = culori.at(labels.at<int>(i, j));
			}
			else {
				imgRez.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			}
		}
	}
	imshow("Colorate", imgRez);
	waitKey(0);

}




int minArray(int arr[], int n)
{
	int minim = arr[0];
	for (int i = 1; i < n; i++)
		minim = min(minim, arr[i]);
	return minim;
}
/*
void douaTrecereri() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;
		int label = 0;

		Mat labels;
		labels = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				labels.at<int>(i, j) = 0;
			}
		}
		vector<vector<int>> edges;
		int di[4] = { -1,0,1,0 };
		int dj[4] = { 0,-1,0,1 };
		uchar neighbors[4];
		vector<int> L;
		int x;
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (src.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0)
				for (int k = 0; k < 4; k++) {
					if (labels.at<uchar>(neighbors[k]) > 0) {
						L.push_back(labels.at<uchar>(neighbors[k]));
					}
					if (L.size() == 0)
					{
						label++;
						labels.at<uchar>(i, j) = label;
					}
					else {
						int nrElem = sizeof(L) / sizeof(L[0]);
						int x;
						//x = minArray(L, nrElem);

						labels.at<uchar>(i, j) = x;
						int dimL = L.size();
						for (int i = 0; i < dimL; i++) {
							int y;
							if (y != x) {
								edges[x].push_back(y);
								edges[y].push_back(x);
							}
						}

					}
				}

			}
		}
		int newLabel = 0;
		int newLabels[] = { 0 };
		int t;
		for ( t = 0;t < label + 1; t++){
			newLabels[t] = 0;
		}
		for (int i = 0; i < label; i++)
		{
			if (newLabels[i] == 0) {
				newLabel++;
				std::queue<int> Q;
				newLabels[i] = newLabel;
				Q.push(i);
				while (!Q.empty()) {
					int x;
					x= Q.front();
					Q.pop();
					for (int y = 1; y <= edges[x].size(); y++) {
						if (newLabels[y] == 0)
						{
							newLabels[y] = newLabel;
							Q.push(y);
						}

					}

				}

			}
		}

		for (int i = 0; i < height - 1; i++) {
			for (int j = 0; j < width - 1; j++) {
				labels.at<uchar>(i, j) = newLabels[labels.at<uchar>(i, j)];
			}
		}

	}

}
*/


typedef struct {
	int x;
	int y;
	int z;
} punct;


void urmarire_contur() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);

		bool pixel_in_region = false;
		int pixel_start1;
		int pixel_start2;
		//dir=7 vecinatate de 8
		int dir = 7;

		//scanam din coltul stanga sus pana se gaseste un pixel
		//care apartine unei regiuni 
		//acest pixel va fi pixelul de start pixel_start 1 / 2 (pt x, y)
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (pixel_in_region = true) {
					if (src.at<uchar>(i, j) == 0) {
						pixel_in_region = true;
						pixel_start1 = i;
						pixel_start2 = j;
					}
				}
			}
		}

		vector <punct> elem_contur;
		int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
		int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };

		elem_contur.push_back(punct{ pixel_start1, pixel_start2, 7 });
		int x = pixel_start1;
		int y = pixel_start2;
		bool ok = false;
		//parcurgere vecinatate de 8
		while (!ok) {
			if (dir % 2 == 0) {
				dir = (dir + 7) % 8;
			}
			else if (dir % 2 == 1) {
				dir = (dir + 6) % 8;
			}
			//se parcurge in continuare in sensul acelor ceasornicului
			while (src.at<uchar>(x + di[dir], y + dj[dir]) != 0) {
				dir = dir + 1;
				if (dir == 8) {
					dir = 0;
				}
			}
			elem_contur.push_back(punct{ x + di[dir], y + dj[dir], (byte)dir });
			x += di[dir];
			y += dj[dir];
			int sz;
			sz = elem_contur.size();
			if (
				//daca elem curent Pn este egal cu al doilea elem P1
				elem_contur.at(sz - 1).x == elem_contur.at(1).x &&
				elem_contur.at(sz - 1).y == elem_contur.at(1).y &&
				//daca elem anterior Pn-1 este egal cu primul element P0
				elem_contur.at(sz - 2).x == elem_contur.at(0).x &&
				elem_contur.at(sz - 2).y == elem_contur.at(0).y &&
				//algoritmul functioneaza pt toate regiunile care au suprafata
				//mai mare de un pixel
				elem_contur.size() > 2) {
				//se incheie algoritmul
				ok = true;
			}
		}
		int c;
		for (int i = 0; i < elem_contur.size(); i++) {
			dst.at<uchar>(elem_contur.at(i).x, elem_contur.at(i).y) = 0;
			if (i > 0) {
				c = (elem_contur.at(i).z - elem_contur.at(i - 1).z + 8) % 8;
				printf("codul inlantuit: %d - derivata codului inlantuita: %d \n", elem_contur.at(i).z, c);
			}
		}
		imshow("image", src);
		imshow("contur imag init", dst);
		waitKey();
	}
}


void urmarire_contur_fisier() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);

		vector <punct> elem_contur;
		int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
		int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };

		bool pixel_in_region = false;
		int pixel_start1;
		int pixel_start2;
		//dir=7 vecinatate de 8
		int dir = 7;
		//scanam din coltul stanga sus pana se gaseste un pixel
		//care apartine unei regiuni 
		//acest pixel va fi pixelul de start pixel_start 1 / 2 (pt x, y)
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (pixel_in_region = true) {
					if (src.at<uchar>(i, j) == 0) {
						pixel_in_region = true;
						pixel_start1 = i;
						pixel_start2 = j;
					}
				}
			}
		}
		int x = pixel_start1;
		int y = pixel_start2;
		elem_contur.push_back(punct{ pixel_start1, pixel_start2, 7 });
		bool ok = false;
		while (!ok) {
			if (dir % 2 == 0) {
				dir = (dir + 7) % 8;
			}
			else if (dir % 2 == 1) {
				dir = (dir + 6) % 8;
			}
			while (src.at<uchar>(x + di[dir], y + dj[dir]) != 0) {
				dir = dir + 1;
				if (dir == 8) {
					dir = 0;
				}
			}
			elem_contur.push_back(punct{ x + di[dir], y + dj[dir], (byte)dir });
			x += di[dir];
			y += dj[dir];
			int n = elem_contur.size();
			if (//daca elem curent Pn este egal cu al doilea elem P1
				elem_contur.at(n - 1).x == elem_contur.at(1).x &&
				elem_contur.at(n - 1).y == elem_contur.at(1).y &&
				//daca elem anterior Pn-1 este egal cu primul element P0
				elem_contur.at(n - 2).x == elem_contur.at(0).x &&
				elem_contur.at(n - 2).y == elem_contur.at(0).y &&
				//algoritmul functioneaza pt toate regiunile care au suprafata
				//mai mare de un pixel
				elem_contur.size() > 2) {
				//se incheie algoritmul
				ok = true;
			}
		}
		FILE* f = fopen("f.txt", "w");
		int c;
		for (int i = 0; i < elem_contur.size(); i++) {
			dst.at<uchar>(elem_contur.at(i).x, elem_contur.at(i).y) = 0;
			if (i > 0) {
				c = (elem_contur.at(i).z - elem_contur.at(i - 1).z + 8) % 8;
				fprintf(f, "codul inlantuita : %d - derivata codului inlantuita: %d \n", elem_contur.at(i).z, c);
			}
		}
		imshow("image", src);
		imshow("contur imag init", dst);
		waitKey();
	}
}





void reconstructie_contur() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		//coord punct start
		int  x;
		int y;
		//dimensiune cod
		int dim;
		//secventa cod
		vector<int> cod;
		FILE* f = fopen("reconstruct.txt", "r");
		fscanf(f, "%d", &x);
		fscanf(f, "%d", &y);
		fscanf(f, "%d", &dim);
		for (int i = 0; i <= dim; i++) {
			fscanf(f, "%d", &cod);
		}
		printf("%d %d %d", x, y, dim);
		for (int i = 0; i <= dim; i++) {
			printf("%d", &cod);
		}
		imshow("src image", src);
		imshow("drs img", dst);
		fclose(f);
		waitKey();
	}
}




//////// lab 7

Mat_<uchar> dilatarea(Mat_<uchar> src) {

	int height = src.rows;
	int width = src.cols;
	Mat dst = Mat(height, width, CV_8UC1);
	int vecini = 8;
	int di[8] = { 0,-1,-1,-1,0,1,1,1 };
	int dj[8] = { 1,1,0,-1,-1,-1,0,1 };

	//initializam matricea laculoarea de fundal
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			dst.at<uchar>(i, j) = 0;
		}
	}

	//Dacă originea elementului 
	//structural coincide cu un punct “obiect” în imagine, 
	//atunci toți pixelii acoperiți de elementul structural 
	//devin pixeli “obiect” în imaginea rezultat.
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (src.at<uchar>(i, j) == 0) {
				for (int r = 0; r < vecini; r++) {
					if (isInside(i + di[r], j + dj[r])) {
						dst.at<uchar>(i + di[r], j + dj[r]) = 0;
					}
				}
			}
		}
	}
	imshow("src image", src);
	imshow("drs img", dst);
	return dst;
	system("pause");
	waitKey(0);
}


void dilatare1() {
	Mat img = imread("C:/Users/Rebeca/Desktop/PI/lab7/Morphological_Op_Images/1_Dilate/mon1thr1_bw.bmp", 0);
	Mat img2(img.rows, img.cols, CV_8UC1);
	for (int i = 1; i < img.rows - 1; i++) {
		for (int j = 1; j < img.cols - 1; j++) {
			if (img.at<uchar>(i, j) == 0)
				img2.at<uchar>(i, j) = 0;
			else if (img.at<uchar>(i, j) == 255) {
				img2.at<uchar>(i - 1, j - 1) = 255;
				img2.at<uchar>(i - 1, j) = 255;
				img2.at<uchar>(i - 1, j + 1) = 255;
				img2.at<uchar>(i, j - 1) = 255;
				img2.at<uchar>(i, j) = 255;
				img2.at<uchar>(i, j + 1) = 255;
				img2.at<uchar>(i + 1, j - 1) = 255;
				img2.at<uchar>(i + 1, j) = 255;
				img2.at<uchar>(i + 1, j + 1) = 255;
			}
		}
	}
	imshow("dilatare", img2);
	waitKey(0);
}

int fun(int i, int j, Mat img) {
	if (img.at<uchar>(i - 1, j - 1) == 0) return 1;
	if (img.at<uchar>(i - 1, j) == 0) return 1;
	if (img.at<uchar>(i - 1, j + 1) == 0)return 1;
	if (img.at<uchar>(i, j - 1) == 0)return 1;
	if (img.at<uchar>(i, j + 1) == 0)return 1;
	if (img.at<uchar>(i + 1, j - 1) == 0)return 1;
	if (img.at<uchar>(i + 1, j) == 0)return 1;
	if (img.at<uchar>(i + 1, j + 1) == 0)return 1;
	return 0;
}

void eroziune1() {
	Mat img = imread("C:/Users/Rebeca/Desktop/PI/lab7/Morphological_Op_Images/2_Erode/mon1_gray.bmp", 0);
	Mat img2(img.rows, img.cols, CV_8UC1);
	for (int i = 1; i < img.rows - 1; i++) {
		for (int j = 1; j < img.cols - 1; j++) {
			if (img.at<uchar>(i, j) == 0)
				img2.at<uchar>(i, j) = 0;
			else if (img.at<uchar>(i, j) == 255) {
				if (fun(i, j, img) == 1)
					img2.at<uchar>(i, j) = 0;
				else
					img2.at<uchar>(i, j) = 255;
			}
		}
	}
	imshow("eroziune", img2);
	waitKey(0);
}

Mat_<uchar> eroziunea(Mat_<uchar> src) {

	int height = src.rows;
	int width = src.cols;
	Mat dst = Mat(height, width, CV_8UC1);
	int vecini = 8;
	int di[8] = { 0,-1,-1,-1,0,1,1,1 };
	int dj[8] = { 1,1,0,-1,-1,-1,0,1 };
	//initializam matricea laculoarea de fundal
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			dst.at<uchar>(i, j) = 0;
		}
	}

	//Dacă elementul structural acoperă cel puțin un punct 
	//de “fundal”, pixelul din imaginea
	//destinație va rămâne pixel de “fundal”.

	for (int i = 0; i < height - 1; i++) {
		for (int j = 0; j < width - 1; j++) {
			if (src.at<uchar>(i, j) == 255) {
				for (int r = 0; r < vecini; r++) {
					if (isInside(i + di[r], j + dj[r])) {
						dst.at<uchar>(i + di[r], j + dj[r]) = 255;
					}
				}
			}
		}
	}
	imshow("src image", src);
	imshow("drs img", dst);
	return dst;
	system("pause");
	waitKey();

}


Mat_<uchar> dechiderea(Mat_<uchar> src) {
	// Mat_<uchar> src;
	int height = src.rows;
	int width = src.cols;
	Mat dst = Mat(height, width, CV_8UC1);
	//Deschiderea constă într-o eroziune urmată de o dilatare
	//este necesară crearea unui buffer de imagine suplimentar.
	Mat_ <uchar> aux = eroziunea(src);
	dst = dilatarea(aux);

	imshow("src image", src);
	imshow("drs img", dst);
	return dst;
	system("pause");
	waitKey();

}

Mat_<uchar> inchiderea(Mat_<uchar> src) {
	//	Mat_<uchar> src;
	int height = src.rows;
	int width = src.cols;
	Mat_<uchar> dst = Mat(height, width, CV_8UC1);

	//Închiderea constă într-o dilatare urmată de o eroziune
	//este necesară crearea unui buffer de imagine suplimentar.
	Mat_ <uchar> aux = dilatarea(src);
	dst = eroziunea(aux);


	imshow("src image", src);
	imshow("drs img", dst);
	return dst;
	system("pause");
	waitKey();

}

void extragere_contur() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		//este necesară crearea unui buffer de imagine suplimentar.
		Mat_<uchar> aux = eroziunea(src);
		//diferenta dintre imaginea src și rezultatul eroziunii ei
		dst = src - aux;
		imshow("src image", src);
		imshow("drs img", dst);
		waitKey();
	}
}

int histLa8()
{
	int hist[256];
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;
		int L = 255; // L = 255 nivelul maxim de intensitate al imaginii
		// h(g) histograma imaginii(numărul de pixeli având nivelul de gri g)
		int M = height * width;// M = H x W, numărul de pixeli din imagine
		int histC[256];

		//calculez si afisez histograma
		for (int i = 0; i < 255; i++) {
			hist[i] = 0;
		}
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				hist[src.at<uchar>(i, j)]++;
			}
		}
		for (int i = 0; i < 255; i++) {
			printf("%d  ", hist[i]);
		}
		showHistogram("histograma", hist, 256, 256);

		//valoarea medie a nivelurilor de intensitate

		float media = 0;
		for (int g = 0; g < 256; g++) {
			//g*h[g]
			media = media + g * hist[g];
		}

		printf("Media2 =%f\n", media / M);

		// deviatia standard
		float deviatia = 0.0;
		for (int g = 0; g < 255; g++) {
			//(g-media)^2*p[g]
			deviatia += pow((g - media), 2) * hist[g];
		}

		printf("Deviatia standard: %f", sqrt(deviatia / M));


		//histograma cumulativa
		for (int i = 1; i < 255; i++) {
			histC[i] = 0;
		}
		for (int i = 1; i < 255; i++) {
			{
				histC[i] = histC[i - 1] + hist[i];
			}
		}
		showHistogram("Histograma cumulativa", histC, 256, 256);

		//Binarizare automată globală

		int min = 0;
		int max = 255;
		//cautam intensitatea maxima siminima din imagine
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (src.at<uchar>(i, j) > max)		max = src.at<uchar>(i, j);
				if (src.at<uchar>(i, j) < min)		min = src.at<uchar>(i, j);
			}
		}
		printf("\nvaloare maxima %d si valoare minima %d", max, min);
		int T = (max + min) / 2;

		//se calculează valoarea medie G1 μ pentru pixelii care satisfac condiția G1: I<T

			//calculez N1
		float N1 = 0.0;
		for (int g = min; g < T; g++) {
			N1 = N1 + hist[g];
		}
		float N2 = 0.0;
		//calculez N2
		for (int g = T + 1; g < max; g++) {
			N2 = N2 + hist[g];
		}
		//calculez media G1
		float G1 = 0.0;
		for (int g = min; g < T; g++) {
			G1 = G1 + g * hist[g];
		}
		float mediaG1 = G1 / N1;

		float G2 = 0.0;
		//calculez media G2
		for (int g = T + 1; g < max; g++) {
			G2 = G2 + g * hist[g];
		}
		float mediaG2 = G2 / N2;

		//actualizează pragul de binarizare
		T = (mediaG1 + mediaG2) / 2;

		//repetă pașii 2-3 până când T T <eroare

	//Se binarizează imaginea folosind pragul T
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {

				if (src.at<uchar>(i, j) < T)
					src.at<uchar>(i, j) = 0;

				if (src.at<uchar>(i, j) > T)
					src.at<uchar>(i, j) = 1;
			}
		}
		imshow("img", src);

		//lățirea / îngustarea histogramei


		//corecția gamma


		waitKey();
	}
	return 0;
}

void negativulImaginii()
{
	//negativului imaginii
	//g out = L - g in = 255 - gin
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = 255 - val;
				dst.at<uchar>(i, j) = neg;
			}
		}

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void cresterea_luminozitatii()
{
	//modificarea luminozității
	//cresterea luminozitatii
	//g out = g in + offset 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = val + 100;
				dst.at<uchar>(i, j) = neg;
				if (neg > 255) neg = 255;
				else if (neg < 0) neg = 0;
			}
		}

		imshow("input image", src);
		imshow("cresterea luminozitatii", dst);
		waitKey();
	}
}

void scaderea_luminozitatii()
{
	//modificarea luminozității
	//scaderea luminozitatii
	//g out = g in - offset 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = val - 100;
				dst.at<uchar>(i, j) = neg;
				if (neg > 255) neg = 255;
				else if (neg < 0) neg = 0;
			}
		}

		imshow("input image", src);
		imshow("scaderea luminozitatii", dst);
		waitKey();
	}
}





/*
Mat_ <uchar> convolutie(Mat_ <uchar> src, Mat_<float> filtru)
{
	int suma1 = 0;
	int suma2 = 0;
	int maxx;
//	Mat_<int> filtru;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{

		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);

		int linii = filtru.rows;
		int coloane = filtru.cols;

		for (int i = 0; i < linii; i++) {
			for (int j = 0; j < coloane; j++) {
				if (filtru(i, j) < 0)
					//suma coef negativi
					suma2 = suma2 + filtru(i, j);
				else
					//suma coef pozitivi
					suma1 = suma1 + filtru(i, j);
			}
		}

		suma2 = -suma2;
		int numarator = 2 * max(suma1, suma2);
		int S = 1 / numarator;
		int L = 255;


		imshow("input image", src);
		imshow("imagine convolutie", dst);
		return dst;
		waitKey();
	}

}*/


// ???????
Mat_ <uchar> convolutie(Mat_ <uchar> src, Mat_<float> filtru)
{
	int suma1 = 0;
	int suma2 = 0;
	int maxx;
	//Mat_<int> filtru;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{

		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);

		int linii = filtru.rows;
		int coloane = filtru.cols;

		for (int i = 0; i < linii; i++) {
			for (int j = 0; j < coloane; j++) {
				if (filtru(i, j) < 0)
					//suma coef negativi
					suma2 = suma2 + filtru(i, j);
				else
					//suma coef pozitivi
					suma1 = suma1 + filtru(i, j);
			}
		}

		suma2 = -suma2;
		int numarator = 2 * max(suma1, suma2);
		int S = 1 / numarator;
		int L = 255;


		imshow("input image", src);
		imshow("imagine convolutie", dst);
		return dst;
		waitKey();
	}

}

void centering_transform(Mat img) {
	// imaginea trebuie să aibă elemente de tip float
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img.at<float>(i, j) = ((i + j) & 1) ? -img.at<float>(i, j) : img.at<float>(i, j);
		}
	}
}

Mat generic_frequency_domain_filter(Mat src) {
	//imaginea trebuie să aibă elemente de tip float
	Mat srcf;
	src.convertTo(srcf, CV_32FC1);
	//transformarea de centrare
	centering_transform(srcf);
	//aplicarea transformatei Fourier, se obține o imagine cu valori numere complexe
	Mat fourier;
	dft(srcf, fourier, DFT_COMPLEX_OUTPUT);
	//divizare în două canale: partea reală și partea imaginară
	Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
	split(fourier, channels); // channels[0] = Re(DFT(I)), channels[1] = Im(DFT(I))
	//calcularea magnitudinii și fazei în imaginile mag, respectiv phi, cu elemente de tip float
	Mat mag, phi;
	magnitude(channels[0], channels[1], mag);
	phase(channels[0], channels[1], phi);
	//aici afișați imaginile cu fazele și magnitudinile
	// ......
	//aici inserați operații de filtrare aplicate pe coeficienții Fourier
	// ......
	//memorați partea reală în channels[0] și partea imaginară în channels[1]
	// ......
	//aplicarea transformatei Fourier inversă și punerea rezultatului în dstf
	Mat dst, dstf;
	merge(channels, 2, fourier);
	dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
	//transformarea de centrare inversă
	centering_transform(dstf);
	//normalizarea rezultatului în imaginea destinație
	normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);
	//Notă: normalizarea distorsionează rezultatul oferind o afișare îmbunătățită în intervalul
	//[0,255]. Dacă se dorește afișarea rezultatului cu exactitate (vezi Activitatea 3) se va
	//folosi în loc de normalizare conversia:
	//dstf.convertTo(dst, CV_8UC1);
	return dst;
}

void gaussian(float w) {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		//Mat g = Mat(w, w, CV_8UC1);
		float a = w / 6.0;

		Mat_<float> g(w, w);

		for (int i = 0; i < w; i++) {
			for (int j = 0; j < w; j++) {
				g(i, j) = 1 / (sqrt(2 * PI) * a * a) * exp(-(pow(i - w / 2, 2) + pow(j - w / 2, 2)) / (2 * a * a));

			}
		}

		dst = convolutie(src, g);

		double t = (double)getTickCount(); // Găsește timpul curent [ms]
		// ... Procesarea propriu-zisă ...
		// Găsește timpul current din nou și calculează timpul scurs [ms]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Afișarea la consolă a timpului de procesare [ms]
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("src", src);
		imshow("dst", dst);
		waitKey();
		//	return dst;
	}

}



void gaussianVectorial(float w)
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		//Mat g = Mat(w, w, CV_8UC1);
		float a = w / 6.0;

		Mat_<float> g(w, w);
		
		vector<int> gx;
		vector<int> gy;
		

		for (int i = 0; i < w; i++) {
			for (int j = 0; j < w; j++) {
				int gx = 1 / (sqrt(2 * PI) * a) * exp(-(pow(i - w / 2, 2) / (2 * a * a)));
				int gy = 1 / (sqrt(2 * PI) * a) * exp(-(pow(j - w / 2, 2) / (2 * a * a)));

			}
		}

		dst = convolutie(src, g);

		double t = (double)getTickCount(); // Găsește timpul curent [ms]
		// ... Procesarea propriu-zisă ...
		// Găsește timpul current din nou și calculează timpul scurs [ms]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Afișarea la consolă a timpului de procesare [ms]
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("src", src);
		imshow("dst", dst);
		waitKey();
		//	return dst;
	}

}
/*
void metoda_canny() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);

		Mat temp = src.clone(); //matrice temporara
		Mat modul = Mat::zeros(src.size(), CV_8UC1); //matricea pt. modulul gradientului
		Mat directie = Mat::zeros(src.size(), CV_8UC1); //matricea pt. directia gradientului

	int Sx[3][3] = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
	int Sy[3][3] = { { -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 } };

	int sigma;
	
	int dir = 0;
	gaussian(6 * sigma);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			int gradX= convolutie(temp.at<float>(i, j), Sx);
			int grasY= convolutie(temp.at<float>(i, j), Sy);
			modul(i, j) = sqrt(gradX * gradX + gradY * gradY) / 5.65;
		}
	}

	float teta = atan2((float)gradY, (float)gradX);
	if ((teta > 3 * PI / 8 && teta < 5 * PI / 8) || (teta > -5 * PI / 8 && teta < -3 * PI / 8)) dir = 0;
	if ((teta > PI / 8 && teta < 3 * PI / 8) || (teta > -7 * PI / 8 && teta < -5 * PI / 8)) dir = 1;
	if ((teta > -PI / 8 && teta < PI / 8) || teta > 7 * PI / 8 && teta < -7 * PI / 8) dir = 2;
	if ((teta > 5 * PI / 8 && teta < 7 * PI / 8) || (teta > -3 * PI / 8 && teta < -PI / 8)) dir = 3;
	directie(i, j) = dir;


	imshow("src", src);
	imshow("dst", dst);
	waitKey();
	
	}

}
*/

void extindere_muchii() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;
		
		Mat temp = src.clone(); //matrice temporara
		Mat modul = Mat::zeros(src.size(), CV_8UC1); //matricea pt. modulul gradientului
		Mat directie = Mat::zeros(src.size(), CV_8UC1); //matricea pt. directia

		//matricea visited 0 (nevizitat) , 1 (vizitat)
		//initializez matricea cu 0
		Mat visited;
		visited = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				visited.at<int>(i, j) = 0;
			}
		}
		int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
		int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
		uchar neighbors[4];

		using namespace std;
		queue <Point> que; //elemente de tip Point pt. memorarea coordonatelor pixelilor
		for (int i = 2; i < height - 3; i++)
		{
			for (int j = 2; j < width - 3; j++)
			{
				if (modul.at<uchar>(i, j) == 0 && visited.at<int>(i, j) == 0)
				{
					que.push(Point(j, i));
					visited.at<int>(i, j) = 1;
					while (!que.empty())
					{
						Point oldest;
						oldest = que.front();
						int ii = oldest.x;
						int jj = oldest.y;
						que.pop();
						for (int k = 0; k < 8; k++)
						{
							neighbors[k] = src.at <uchar>(ii + di[k], jj + dj[k]);
							if (modul.at<uchar>(neighbors[k]) ==0) {

								visited.at<int>(neighbors[k]) = 1;
								que.push(Point(neighbors[k]));
							}
						}
					}
				}
			}
		}
		// Generare paleta de culori pentru afisarea directiilor gradientului
		Scalar colorLUT[4] = { 0 };
		colorLUT[0] = Scalar(0, 0, 255); //red
		colorLUT[1] = Scalar(0, 255, 255); // yellow
		colorLUT[2] = Scalar(255, 0, 0); // blue
		colorLUT[3] = Scalar(0, 255, 0); // green
		Mat_<Vec3b> ImgDir = Mat::zeros(src.size(), CV_8UC3);
		int d = 0;
		for (int i = d; i < height - d; i++) // d=1
			for (int j = d; j < width - d; j++)
				if (modul.at<int>(i, j))
				{
					Scalar color = colorLUT[directie.at<int>(i, j)];
					ImgDir(i, j)[0] = color[0];
					ImgDir(i, j)[1] = color[1];
					ImgDir(i, j)[2] = color[2];
				}

		imshow("Imagine directii", ImgDir);

	}
}
/*
void gaussian_filter(Mat_<uchar> img, Mat_<uchar> dst) {
	int w = 3;

	float sigma = (float)w / 6.0;
	Mat g = Mat(w, w, CV_32FC1);

	float ct = 1.0 / (2 * PI * sigma * sigma);

	for (int i = 0; i < w; i++) {
		for (int j = 0; j < w; j++) {
			g.at<float>(i, j) = ct * exp((-((i - w / 2) * (i - w / 2) + (j - w / 2) * (j - w / 2))) / (2 * sigma * sigma));
		}
	}

	convolution(g, img, dst);

}

void gradient_and_direction(Mat_<uchar> img, Mat sobel, Mat phi, Mat mag) {
	int fx_values[9] = { -1,-0,1,-2,0,2,-1,0,1 };
	Mat_<int> fx = Mat_<int>(3, 3, fx_values);
	int fy_values[9] = { 1,2,1,0,0,0,-1,-2,-1 };
	Mat_<int> fy = Mat_<int>(3, 3, fy_values);

	Mat_<float> x, y;

	Mat Gx = Mat(img.rows, img.cols, CV_32FC1);
	Mat Gy = Mat(img.rows, img.cols, CV_32FC1);

	for (int i = 1; i < img.rows - 1; i++)
	{
		for (int j = 1; j < img.cols - 1; j++)
		{
			Gx.at<float>(i, j) = 0;
			Gy.at<float>(i, j) = 0;

			for (int u = 0; u < 3; u++)
			{
				for (int v = 0; v < 3; v++)
				{
					Gx.at<float>(i, j) += fx.at<int>(u, v) * img.at<uchar>(i - 1 + u, j - 1 + v);
					Gy.at<float>(i, j) += fy.at<int>(u, v) * img.at<uchar>(i - 1 + u, j - 1 + v);
				}
			}

			mag.at<float>(i, j) = (float)sqrt(pow(Gx.at<float>(i, j), 2) + (pow(Gy.at<float>(i, j), 2))) / 4 * sqrt(2);
			phi.at<float>(i, j) = atan2(Gy.at<float>(i, j), Gx.at<float>(i, j));
			sobel.at<uchar>(i, j) = (uchar)mag.at<float>(i, j);
		}
	}
}

void suppression(Mat img, Mat dst, Mat phi, Mat mag) {
	for (int i = 1; i < img.rows - 1; i++) {
		for (int j = 1; j < img.cols - 1; j++) {
			//slice 0
			if ((phi.at<float>(i, j) >= -CV_PI / 8 && phi.at<float>(i, j) <= CV_PI / 8) || (phi.at<float>(i, j) >= 7 * CV_PI / 8) || (phi.at<float>(i, j) <= -7 * CV_PI / 8))
			{
				if (mag.at<float>(i, j) >= mag.at<float>(i, j - 1) && mag.at<float>(i, j) >= mag.at<float>(i, j + 1))
					dst.at<uchar>(i, j) = (uchar)mag.at<float>(i, j);
				else
					dst.at<uchar>(i, j) = 0;
			}

			//slice 1
			else if ((phi.at<float>(i, j) >= CV_PI / 8 && phi.at<float>(i, j) <= 3 * CV_PI / 8) || (phi.at<float>(i, j) >= -7 * CV_PI / 8 && phi.at<float>(i, j) <= -5 * CV_PI / 8))
			{
				if (mag.at<float>(i, j) >= mag.at<float>(i - 1, j + 1) && mag.at<float>(i, j) >= mag.at<float>(i + 1, j - 1))
					dst.at<uchar>(i, j) = (uchar)mag.at<float>(i, j);
				else
					dst.at<uchar>(i, j) = 0;
			}

			//slice 2
			else if ((phi.at<float>(i, j) >= 3 * CV_PI / 8 && phi.at<float>(i, j) <= 5 * CV_PI / 8) || (phi.at<float>(i, j) >= -5 * CV_PI / 8 && phi.at<float>(i, j) <= -3 * CV_PI / 8))
			{
				if (mag.at<float>(i, j) >= mag.at<float>(i - 1, j) && mag.at<float>(i, j) >= mag.at<float>(i + 1, j))
					dst.at<uchar>(i, j) = (uchar)mag.at<float>(i, j);
				else
					dst.at<uchar>(i, j) = 0;
			}

			//slice 3
			else if ((phi.at<float>(i, j) >= 5 * CV_PI / 8 && phi.at<float>(i, j) <= 7 * CV_PI / 8) || (phi.at<float>(i, j) >= -3 * CV_PI / 8 && phi.at<float>(i, j) <= -CV_PI / 8))
			{
				if (mag.at<float>(i, j) >= mag.at<float>(i - 1, j - 1) && mag.at<float>(i, j) >= mag.at<float>(i + 1, j + 1))
					dst.at<uchar>(i, j) = (uchar)mag.at<float>(i, j);
				else
					dst.at<uchar>(i, j) = 0;
			}
		}
	}
}

void adaptive_thresholding(Mat mag, Mat dst) {
	int* hist = (int*)calloc(sizeof(int), 256);
	histLa8(mag, hist);

	float p = 0.1;
	float k = 0.4;

	float nr_non_edge_pixels = (1 - p) * ((mag.rows - 2) * (mag.cols - 2) - hist[0]);
	float threshold_high = 0;

	int sum_hist = hist[1];
	for (int i = 2; i < 256; i++) {
		if (sum_hist >= nr_non_edge_pixels) {
			threshold_high = i;
			break;
		}
		sum_hist += hist[i];
	}

	float threshold_low = k * threshold_high;

	uchar NO_EDGE = 0, WEAK_EDGE = 128, STRONG_EDGE = 255;

	for (int i = 0; i < mag.rows; i++) {
		for (int j = 0; j < mag.cols; j++) {
			if (mag.at<uchar>(i, j) < threshold_low) {
				dst.at<uchar>(i, j) = NO_EDGE;
			}
			if (mag.at<uchar>(i, j) >= threshold_low && mag.at<uchar>(i, j) <= threshold_high) {
				dst.at<uchar>(i, j) = WEAK_EDGE;
			}
			if (mag.at<uchar>(i, j) > threshold_high) {
				dst.at<uchar>(i, j) = STRONG_EDGE;
			}
		}
	}
}

void edge_linking(Mat adaptive, Mat canny) {

	identitate(adaptive, canny);

	uchar NO_EDGE = 0, WEAK_EDGE = 128, STRONG_EDGE = 255;

	queue<Point> q;

	int dj[8] = { -1, -1, -1, 0, 1, 1, 1, 0 };
	int di[8] = { -1, 0, 1, 1, 1, 0, -1, -1 };

	for (int i = 0; i < adaptive.rows; i++) {
		for (int j = 0; j < adaptive.cols; j++) {
			if (canny.at<uchar>(i, j) == STRONG_EDGE) {
				Point p = Point(i, j);
				q.push(p);
			}
			while (!q.empty()) {
				int x, y;
				Point first = q.front();
				q.pop();
				int ni, nj;
				for (int k = 0; k < 8; k++) {
					//for each neighbour
					ni = first.x + di[k];
					nj = first.y + dj[k];
					if (ni < adaptive.rows && ni >= 0 && nj < adaptive.cols && nj >= 0) {
						if (canny.at<uchar>(ni, nj) == WEAK_EDGE) {
							canny.at<uchar>(ni, nj) = STRONG_EDGE;
							q.push({ ni,nj });
						}
					}
				}

			}
		}
	}

	for (int i = 0; i < adaptive.rows; i++) {
		for (int j = 0; j < adaptive.cols; j++) {
			if (canny.at<uchar>(i, j) == WEAK_EDGE) {
				canny.at<uchar>(i, j) = 0;
			}
		}
	}

}

void canny_method() {
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat_<uchar> img = imread(fname, IMREAD_GRAYSCALE);
	Mat_<uchar> dst_aux = Mat(img.rows, img.cols, CV_8UC1, Scalar(255, 255, 255));
	Mat_<uchar> dst = Mat(img.rows, img.cols, CV_8UC1, Scalar(255, 255, 255));
	Mat phi = Mat(img.rows, img.cols, CV_32FC1);
	Mat mag = Mat(img.rows, img.cols, CV_32FC1);
	Mat supp = Mat(img.rows, img.cols, CV_8UC1);
	Mat adaptive = Mat(img.rows, img.cols, CV_8UC1);
	Mat canny = Mat(img.rows, img.cols, CV_8UC1);

	gaussian_filter(img, dst_aux);
	gradient_and_direction(dst_aux, dst, phi, mag);
	suppression(img, supp, phi, mag);
	adaptive_thresholding(supp, adaptive);
	edge_linking(adaptive, canny);

	imshow("Initial", img);
	imshow("Filtru Gaussian", dst_aux);
	imshow("Gradient", dst);
	imshow("Suppression", supp);
	imshow("Adaptive Thresholding", adaptive);
	imshow("Canny Edge", canny);
	waitKey(0);
}

*/
int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - Grey - addition factor\n");
		printf(" 11 - Grey - multiplication factor\n");
		printf(" 12 - Create color image\n");
		printf(" 13 - Matrice\n");
		printf(" 14 - CopyRGB\n");
		printf(" 15 - RGB to greyScale\n");
		printf(" 16 - grey scale to binary\n");
		printf(" 17 - RGB to HSV\n");
		printf(" 18 - is inside \n");
		printf(" 19 - Calc vector histograma \n");
		printf(" 20 - Normalizare histograma \n");
		//printf(" 21 - Show histograma \n");
		printf(" 21 - Aria \n");
		printf(" 22 - Centrul de masa \n");
		printf("23 - Traversare latime\n");
		printf("24 - Contur\n");
		printf("25 - Contur scris in fisier\n");
		printf("26 - Reconstructie contur din fisier\n");
		printf("27 - Dilatarea\n");
		printf("28 - Eroziunea\n");
		printf("29 - Deschiderea\n");
		printf("30 - Inchiderea\n");
		printf("31 - Histograma lab 8\n");
		printf("32 - Cresterea luminozitatii\n");
		printf("33 - Scaderea luminozitatii\n");
		printf("34- Convolutie\n");
		printf("35 - gaussian\n");
		printf("36 - lab 4\n");
		printf("37 - parcurgere latime\n");
		printf("38 - negativul imaginii\n");
		printf("39 - dilatarea imaginii\n");
		printf("40 - eroziunea imaginii\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);

		Mat_ <uchar> src = imread("C:/Users/Rebeca/Desktop/PI/cameraman.bmp", IMREAD_GRAYSCALE);

		Mat_ <uchar> img = imread("C:/Users/Rebeca/Desktop/PI/cameraman.bmp", IMREAD_GRAYSCALE);
		int k[9] = { 0, -1, 0, -1 ,4, -1, 0 ,-1 ,0 };
		Mat_<int> filtru(3, 3, k);
		filtru.setTo(1);

		Mat_ <uchar> img4 = imread("C:/Users/Rebeca/Desktop/PI/portrait_Gauss1.bmp", IMREAD_GRAYSCALE);

		switch (op)
		{
		case 1:
			testOpenImage();
			break;
		case 2:
			testOpenImagesFld();
			break;
		case 3:
			testParcurgereSimplaDiblookStyle(); //diblook style
			break;
		case 4:
			//testColor2Gray();
			testBGR2HSV();
			break;
		case 5:
			testResize();
			break;
		case 6:
			testCanny();
			break;
		case 7:
			testVideoSequence();
			break;
		case 8:
			testSnap();
			break;
		case 9:
			testMouseClick();
			break;
		case 10:
			testGreyPlus();
			break;
		case 11:
			testGreyMultiplication();
			break;
		case 12:
			createCologImg();
			break;
		case 13:
			matriceInversa();
			break;
		case 14:
			copyRGBcolors();
			break;
		case 15:
			rgbToGrey();
			break;
		case 16:
			greyScaleToBinary();
			break;
		case 17:
			rgbToHSV();
			break;
		case 18:
			int a, b;
			scanf("%d%d", &a, &b);
			if (isInside(a, b)) printf("Outside\n"); else printf("Inside\n");
			system("pause");
			break;
		case 19:
			calcHist();
			break;
		case 20:
			normHist();
			break;

		case 22:
			centru_masa();
			break;
		
		case 24:
			urmarire_contur();
			break;
		case 25:
			urmarire_contur_fisier();
			break;
		case 26:
			reconstructie_contur();
			break;
		case 27:

			src = openImg();
			dilatarea(src);
			break;
		case 28:
			src = openImg();
			eroziunea(src);
			break;
		case 29:
			src = openImg();
			dechiderea(src);
			break;
		case 30:
			src = openImg();
			inchiderea(src);
			break;
		case 31:
			histLa8();
			break;
		case 32:


			cresterea_luminozitatii();
			break;
		case 33:

			scaderea_luminozitatii();
			break;

		case 34:
			convolutie(img, filtru);
			break;
		case 35:
			gaussian(7);
			break;
		case 36:
			laborator4();
			break;
		case 37:
			traversare_latime();
			break;
		case 38:
			negativulImaginii();
			break;
		case 39:
			dilatare1();
			break;
		case 40:
			dilatare1();
			break;





		}
	} while (op != 0);
	return 0;
}