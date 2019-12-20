#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/ximgproc/disparity_filter.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>



#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>



using namespace cv;
using namespace cv::ximgproc;
using namespace std;

Rect computeROI(Size2i src_sz, Ptr<StereoMatcher> matcher_instance);

const String keys =
    "{help h usage ? |                  | print this message                                                }"
    "{@left          |bL5.jpg | left view of the stereopair                                       }"
    "{@right         |bR5.jpg | right view of the stereopair                                      }"
    "{GT             |../data/aloeGT.png| optional ground-truth disparity (MPI-Sintel or Middlebury format) }"
    "{dst_path       |None              | optional path to save the resulting filtered disparity map        }"
    "{dst_raw_path   |None              | optional path to save raw disparity map before filtering          }"
    "{algorithm      |bm                | stereo matching method (bm or sgbm)                               }"
    "{filter         |wls_conf          | used post-filtering (wls_conf or wls_no_conf)                     }"
    "{no-display     |                  | don't display results                                             }"
    "{no-downscale   |                  | force stereo matching on full-sized views to improve quality      }"
    "{dst_conf_path  |None              | optional path to save the confidence map used in filtering        }"
    "{vis_mult       |1.0               | coefficient used to scale disparity map visualizations            }"
    "{max_disparity  |160               | parameter of stereo matching                                      }"
    "{window_size    |-1                | parameter of stereo matching                                      }"
    "{wls_lambda     |8000.0            | parameter of post-filtering                                       }"
    "{wls_sigma      |1.5               | parameter of post-filtering                                       }"
    ;



void exportMesh(cv::Mat Depth, cv::Mat Normals, cv::Mat texture) {

  std::ofstream objFile, mtlFile;
  objFile.open("export.obj");
  int width = Depth.cols;
  int height = Depth.rows;

  objFile << "mtllib export.mtl" << std::endl;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      objFile << "v " << x << " " << y << " "
              << Depth.at<float>(cv::Point(x, y)) << std::endl;
      objFile << "vt " << x / (width - 1.0f) << " " << (1.0f - y) / height
              << " "
              << "0.0" << std::endl;
      objFile << "vn " << (float)Normals.at<cv::Vec3b>(y, x)[0] << " "
              << (float)Normals.at<cv::Vec3b>(y, x)[1] << " "
              << (float)Normals.at<cv::Vec3b>(y, x)[2] << std::endl;
    }
  }


  objFile << "usemtl picture" << std::endl;

  for (int y = 0; y < height - 1; y++) {
    for (int x = 0; x < width - 1; x++) {
      int f1 = x + y * width + 1;
      int f2 = x + y * width + 2;
      int f3 = x + (y + 1) * width + 1;
      int f4 = x + (y + 1) * width + 2;
      objFile << "f " << f1 << "/" << f1 << "/" << f1 << " ";
      objFile << f2 << "/" << f2 << "/" << f2 << " ";
      objFile << f3 << "/" << f3 << "/" << f3 << std::endl;
      objFile << "f " << f2 << "/" << f2 << "/" << f2 << " ";
      objFile << f4 << "/" << f4 << "/" << f4 << " ";
      objFile << f3 << "/" << f3 << "/" << f3 << std::endl;
    }
  }

  cv::imwrite("export.jpg", texture);
  mtlFile.open("export.mtl");
  mtlFile << "newmtl picture" << std::endl;
  mtlFile << "map_Kd export.jpg" << std::endl;
  objFile.close();
  mtlFile.close();
}



int main(int argc, char** argv)
{
    CommandLineParser parser(argc,argv,keys);
    parser.about("Disparity Filtering Demo");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    String left_im = parser.get<String>(0);
    String right_im = parser.get<String>(1);
    String GT_path = parser.get<String>("GT");

    String dst_path = parser.get<String>("dst_path");
    String dst_raw_path = parser.get<String>("dst_raw_path");
    String dst_conf_path = parser.get<String>("dst_conf_path");
    String algo = parser.get<String>("algorithm");
    String filter = parser.get<String>("filter");
    bool no_display = parser.has("no-display");
    bool no_downscale = parser.has("no-downscale");
    int max_disp = parser.get<int>("max_disparity");
    double lambda = parser.get<double>("wls_lambda");
    double sigma  = parser.get<double>("wls_sigma");
    double vis_mult = parser.get<double>("vis_mult");

    int wsize;
    if(parser.get<int>("window_size")>=0) //user provided window_size value
        wsize = parser.get<int>("window_size");
    else
    {
        if(algo=="sgbm")
            wsize = 3; //default window size for SGBM
        else if(!no_downscale && algo=="bm" && filter=="wls_conf")
            wsize = 7; //default window size for BM on downscaled views (downscaling is performed only for wls_conf)
        else
            wsize = 15; //default window size for BM on full-sized views
    }

    if (!parser.check())
    {
        parser.printErrors();
        return -1;
    }

    //! [load_views]
    Mat left  = imread(left_im ,IMREAD_COLOR);
    if ( left.empty() )
    {
        cout<<"Cannot read image file: "<<left_im;
        return -1;
    }

    Mat right = imread(right_im,IMREAD_COLOR);
    if ( right.empty() )
    {
        cout<<"Cannot read image file: "<<right_im;
        return -1;
    }
    //! [load_views]

    bool noGT;
    Mat GT_disp;
    if (GT_path=="../data/aloeGT.png" && left_im!="../data/aloeL.jpg")
        noGT=true;
    else
    {
        noGT=false;
        if(readGT(GT_path,GT_disp)!=0)
        {
            cout<<"Cannot read ground truth image file: "<<GT_path<<endl;
            return -1;
        }
    }

    Mat left_for_matcher, right_for_matcher;
    Mat left_disp,right_disp;
    Mat filtered_disp;
    Mat conf_map = Mat(left.rows,left.cols,CV_8U);
    conf_map = Scalar(255);
    Rect ROI;
    Ptr<DisparityWLSFilter> wls_filter;
    double matching_time, filtering_time;
    if(max_disp<=0 || max_disp%16!=0)
    {
        cout<<"Incorrect max_disparity value: it should be positive and divisible by 16";
        return -1;
    }
    if(wsize<=0 || wsize%2!=1)
    {
        cout<<"Incorrect window_size value: it should be positive and odd";
        return -1;
    }
    if(filter=="wls_conf") // filtering with confidence (significantly better quality than wls_no_conf)
    {
        if(!no_downscale)
        {
            // downscale the views to speed-up the matching stage, as we will need to compute both left
            // and right disparity maps for confidence map computation
            //! [downscale]
            max_disp/=2;
            if(max_disp%16!=0)
                max_disp += 16-(max_disp%16);
            resize(left ,left_for_matcher ,Size(),0.5,0.5, CV_INTER_LINEAR);
            resize(right,right_for_matcher,Size(),0.5,0.5, CV_INTER_LINEAR);
            //! [downscale]
        }
        else
        {
            left_for_matcher  = left.clone();
            right_for_matcher = right.clone();
        }

        if(algo=="bm")
        {
            //! [matching]
            Ptr<StereoBM> left_matcher = StereoBM::create(max_disp,wsize);
            wls_filter = createDisparityWLSFilter(left_matcher);
            Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);

            cvtColor(left_for_matcher,  left_for_matcher,  COLOR_BGR2GRAY);
            cvtColor(right_for_matcher, right_for_matcher, COLOR_BGR2GRAY);

            matching_time = (double)getTickCount();
            left_matcher-> compute(left_for_matcher, right_for_matcher,left_disp);
            right_matcher->compute(right_for_matcher,left_for_matcher, right_disp);
            matching_time = ((double)getTickCount() - matching_time)/getTickFrequency();
            //! [matching]
        }
        else if(algo=="sgbm")
        {
            Ptr<StereoSGBM> left_matcher  = StereoSGBM::create(0,max_disp,wsize);
            left_matcher->setP1(24*wsize*wsize);
            left_matcher->setP2(96*wsize*wsize);
            left_matcher->setPreFilterCap(63);
            left_matcher->setMode(StereoSGBM::MODE_SGBM_3WAY);
            wls_filter = createDisparityWLSFilter(left_matcher);
            Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);

            matching_time = (double)getTickCount();
            left_matcher-> compute(left_for_matcher, right_for_matcher,left_disp);
            right_matcher->compute(right_for_matcher,left_for_matcher, right_disp);
            matching_time = ((double)getTickCount() - matching_time)/getTickFrequency();
        }
        else
        {
            cout<<"Unsupported algorithm";
            return -1;
        }

        //! [filtering]
        wls_filter->setLambda(lambda);
        wls_filter->setSigmaColor(sigma);
        filtering_time = (double)getTickCount();
        wls_filter->filter(left_disp,left,filtered_disp,right_disp);
        filtering_time = ((double)getTickCount() - filtering_time)/getTickFrequency();
        //! [filtering]
        conf_map = wls_filter->getConfidenceMap();

        // Get the ROI that was used in the last filter call:
        ROI = wls_filter->getROI();
        if(!no_downscale)
        {
            // upscale raw disparity and ROI back for a proper comparison:
            resize(left_disp,left_disp,Size(),2.0,2.0,CV_INTER_LINEAR);
            left_disp = left_disp*2.0;
            ROI = Rect(ROI.x*2,ROI.y*2,ROI.width*2,ROI.height*2);
        }
    }
    else if(filter=="wls_no_conf")
    {
        /* There is no convenience function for the case of filtering with no confidence, so we
        will need to set the ROI and matcher parameters manually */

        left_for_matcher  = left.clone();
        right_for_matcher = right.clone();

        if(algo=="bm")
        {
            Ptr<StereoBM> matcher  = StereoBM::create(max_disp,wsize);
            matcher->setTextureThreshold(0);
            matcher->setUniquenessRatio(0);
            cvtColor(left_for_matcher,  left_for_matcher, COLOR_BGR2GRAY);
            cvtColor(right_for_matcher, right_for_matcher, COLOR_BGR2GRAY);
            ROI = computeROI(left_for_matcher.size(),matcher);
            wls_filter = createDisparityWLSFilterGeneric(false);
            wls_filter->setDepthDiscontinuityRadius((int)ceil(0.33*wsize));

            matching_time = (double)getTickCount();
            matcher->compute(left_for_matcher,right_for_matcher,left_disp);
            matching_time = ((double)getTickCount() - matching_time)/getTickFrequency();
        }
        else if(algo=="sgbm")
        {
            Ptr<StereoSGBM> matcher  = StereoSGBM::create(0,max_disp,wsize);
            matcher->setUniquenessRatio(0);
            matcher->setDisp12MaxDiff(1000000);
            matcher->setSpeckleWindowSize(0);
            matcher->setP1(24*wsize*wsize);
            matcher->setP2(96*wsize*wsize);
            matcher->setMode(StereoSGBM::MODE_SGBM_3WAY);
            ROI = computeROI(left_for_matcher.size(),matcher);
            wls_filter = createDisparityWLSFilterGeneric(false);
            wls_filter->setDepthDiscontinuityRadius((int)ceil(0.5*wsize));

            matching_time = (double)getTickCount();
            matcher->compute(left_for_matcher,right_for_matcher,left_disp);
            matching_time = ((double)getTickCount() - matching_time)/getTickFrequency();
        }
        else
        {
            cout<<"Unsupported algorithm";
            return -1;
        }

        wls_filter->setLambda(lambda);
        wls_filter->setSigmaColor(sigma);
        filtering_time = (double)getTickCount();
        wls_filter->filter(left_disp,left,filtered_disp,Mat(),ROI);
        filtering_time = ((double)getTickCount() - filtering_time)/getTickFrequency();
    }
    else
    {
        cout<<"Unsupported filter";
        return -1;
    }

    //collect and print all the stats:
    cout.precision(2);
    cout<<"Matching time:  "<<matching_time<<"s"<<endl;
    cout<<"Filtering time: "<<filtering_time<<"s"<<endl;
    cout<<endl;

    double MSE_before,percent_bad_before,MSE_after,percent_bad_after;
    if(!noGT)
    {
        MSE_before = computeMSE(GT_disp,left_disp,ROI);
        percent_bad_before = computeBadPixelPercent(GT_disp,left_disp,ROI);
        MSE_after = computeMSE(GT_disp,filtered_disp,ROI);
        percent_bad_after = computeBadPixelPercent(GT_disp,filtered_disp,ROI);

        cout.precision(5);
        cout<<"MSE before filtering: "<<MSE_before<<endl;
        cout<<"MSE after filtering:  "<<MSE_after<<endl;
        cout<<endl;
        cout.precision(3);
        cout<<"Percent of bad pixels before filtering: "<<percent_bad_before<<endl;
        cout<<"Percent of bad pixels after filtering:  "<<percent_bad_after<<endl;
    }

    if(dst_path!="None")
    {
        Mat filtered_disp_vis;
        getDisparityVis(filtered_disp,filtered_disp_vis,vis_mult);
        imwrite(dst_path,filtered_disp_vis);
    }
    if(dst_raw_path!="None")
    {
        Mat raw_disp_vis;
        getDisparityVis(left_disp,raw_disp_vis,vis_mult);
        imwrite(dst_raw_path,raw_disp_vis);
    }
    if(dst_conf_path!="None")
    {
        imwrite(dst_conf_path,conf_map);
    }

    if(!no_display)
    {
        namedWindow("left", WINDOW_AUTOSIZE);
        //imshow("left", left);
        namedWindow("right", WINDOW_AUTOSIZE);
        //imshow("right", right);

        if(!noGT)
        {
            Mat GT_disp_vis;
            getDisparityVis(GT_disp,GT_disp_vis,vis_mult);
            namedWindow("ground-truth disparity", WINDOW_AUTOSIZE);
            //imshow("ground-truth disparity", GT_disp_vis);
        }
vector<int> compression_params;
compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
compression_params.push_back(CV_32FC3);
        //! [visualization]
        Mat raw_disp_vis;
        getDisparityVis(left_disp,raw_disp_vis,vis_mult);
        namedWindow("raw disparity", WINDOW_AUTOSIZE);
//        imshow("raw disparity", raw_disp_vis);
        imwrite("disparity.png", raw_disp_vis, compression_params);
        Mat filtered_disp_vis;
        getDisparityVis(filtered_disp,filtered_disp_vis,vis_mult);
        namedWindow("filtered disparity", WINDOW_AUTOSIZE);
        imshow("filtered disparity", filtered_disp_vis);
        imwrite("filtered_disparity.png", filtered_disp_vis, compression_params);
        if(filtered_disp_vis.type() != CV_32FC1)
        filtered_disp_vis.convertTo(filtered_disp_vis, CV_32FC1);

Mat normals(filtered_disp_vis.size(), CV_32FC3);

for(int x = 0; x < filtered_disp_vis.rows; ++x)
{
    for(int y = 0; y < filtered_disp_vis.cols; ++y)
    {
        // use float instead of double otherwise you will not get the correct result
        // check my updates in the original post. I have not figure out yet why this
        // is happening.
        float dzdx = (filtered_disp_vis.at<float>(x+1, y) - filtered_disp_vis.at<float>(x-1, y)) / 2.0;
        float dzdy = (filtered_disp_vis.at<float>(x, y+1) - filtered_disp_vis.at<float>(x, y-1)) / 2.0;

        Vec3f d(-dzdx, -dzdy, 1.0f);

        Vec3f n = normalize(d);
        normals.at<Vec3f>(x, y) = n;
    }
}
cout<<"done";
imshow("normals", normals);
Mat3b imageF_8UC3;
normals.convertTo(imageF_8UC3, CV_8UC3, 255);
imwrite("test.png", imageF_8UC3);
//imwrite("normal1.png", normals, compression_params);
Mat normals2(filtered_disp_vis.size(), CV_32FC3);

for(int x = 1; x < filtered_disp_vis.cols - 1; ++x)
{
    for(int y = 1; y < filtered_disp_vis.rows - 1; ++y)
    {
            // 3d pixels, think (x,y, filtered_disp_vis)
             /* * * * *
              * * t * *
              * l c * *
              * * * * */
        Vec3f t(x,y-1,filtered_disp_vis.at<float>(y-1, x));
        Vec3f l(x-1,y,filtered_disp_vis.at<float>(y, x-1));
        Vec3f c(x,y,filtered_disp_vis.at<float>(y, x));

        Vec3f d = (l-c).cross(t-c);

        Vec3f n = normalize(d);
        normals2.at<Vec3f>(y,x) = n;
    }
}

//imshow("explicitly cross_product normals2", normals2);
Mat3b imageF_8UC32;
normals2.convertTo(imageF_8UC32, CV_8UC3, 255);
imwrite("test2.png", imageF_8UC3);
//imwrite("normal2.png", normals2, compression_params);
Mat texture;
texture= cv::imread("bL5.jpg" , CV_LOAD_IMAGE_COLOR);
//texture.convertTo(texture, CV_32FC1);
//imshow("texture",texture);
exportMesh(filtered_disp_vis,normals,texture);
waitKey();

        //! [visualization]
    }

    return 0;
}

Rect computeROI(Size2i src_sz, Ptr<StereoMatcher> matcher_instance)
{
    int min_disparity = matcher_instance->getMinDisparity();
    int num_disparities = matcher_instance->getNumDisparities();
    int block_size = matcher_instance->getBlockSize();

    int bs2 = block_size/2;
    int minD = min_disparity, maxD = min_disparity + num_disparities - 1;

    int xmin = maxD + bs2;
    int xmax = src_sz.width + minD - bs2;
    int ymin = bs2;
    int ymax = src_sz.height - bs2;

    Rect r(xmin, ymin, xmax - xmin, ymax - ymin);
    return r;
}
