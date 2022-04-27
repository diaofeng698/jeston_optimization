#include <dirent.h>
#include <opencv2/opencv.hpp>

static inline cv::Mat preprocess_img(cv::Mat &img, int input_w, int input_h)
{
    // int w, h, x, y;
    // float r_w = input_w / (img.cols * 1.0);
    // float r_h = input_h / (img.rows * 1.0);
    // if (r_h > r_w)
    // {
    //     w = input_w;
    //     h = r_w * img.rows;
    //     x = 0;
    //     y = (input_h - h) / 2;
    // }
    // else
    // {
    //     w = r_h * img.cols;
    //     h = input_h;
    //     x = (input_w - w) / 2;
    //     y = 0;
    // }
    // cv::Mat re(h, w, CV_8UC3);
    // cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    // cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    // re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    // std::cout << "raw inputH " << img.rows << " inputW " << img.cols << " inputChannel " << img.channels() <<
    // std::endl; std::cout << "raw inputH " << input_w << " inputW " << input_h << std::endl;
    cv::Mat gray, resized_img;

    cv::resize(img, resized_img, cv::Size(input_h, input_w));
    // std::cout << "resized inputH " << resized_img.rows << " inputW " << resized_img.cols << " inputChannel "
    //           << resized_img.channels() << std::endl;

    cv::cvtColor(resized_img, gray, cv::COLOR_BGR2GRAY);
    // std::cout << "gray inputH " << gray.rows << " inputW " << gray.cols << " inputChannel " << gray.channels()
    //           << std::endl;

    // cv::cvtColor(gray, out, cv::COLOR_GRAY2BGR);
    // std::cout << "out inputH " << out.rows << " inputW " << out.cols << " inputChannel " << out.channels() <<
    // std::endl;
    cv::Mat out(input_w, input_h, CV_32FC3, cv::Scalar(0, 0, 0));
    for (int i = 0; i < gray.rows; i++)
    {
        uchar *ptr = gray.ptr<uchar>(i);
        for (int j = 0; j < gray.cols * gray.channels(); j++)
        {
            cv::Vec3f vc3;
            vc3.val[0] = vc3.val[1] = vc3.val[2] = (uchar)ptr[j];

            out.at<cv::Vec3f>(i, j) = vc3;
        }
    }

    // cv::imshow("video", out);
    // cv::waitKey(0);

    return out;
}

static inline int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names, int classesNum)
{

    for (int i = 0; i < classesNum; i++)
    {
        std::string dir_name = p_dir_name + std::to_string(i) + '/';
        // std::cout << "------>   Step into " << dir_name << std::endl;
        // std::cout << dir_name << std::endl;
        DIR *p_dir = opendir(dir_name.c_str());
        if (p_dir == nullptr)
        {
            return -1;
        }

        struct dirent *p_file = nullptr;
        while ((p_file = readdir(p_dir)) != nullptr)
        {
            if (strcmp(p_file->d_name, ".") != 0 && strcmp(p_file->d_name, "..") != 0)
            {
                std::string cur_file_name(dir_name);
                // cur_file_name += "/";
                cur_file_name += p_file->d_name;
                // std::string cur_file_name(p_file->d_name);
                file_names.push_back(cur_file_name);
            }
        }

        std::cout << "------>   Image num " << file_names.size() << std::endl;

        closedir(p_dir);
    }
    return 0;
}
