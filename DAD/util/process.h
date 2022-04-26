#pragma once
#include <opencv2/opencv.hpp>

bool resize_image(const void *src, int src_wdith, int src_height, int width_step, void *dst_buf, int dst_width,
                  int dst_height);

bool padding_image(const void *src, int src_wdith, int src_height, int width_step, unsigned char pad_vaule[3],
                   void *dst_buf, int dst_wid, int dst_hei);

bool normlize_image(float *src, int src_wdith, int src_height, int channel, float mean[3], float std_val[3]);

bool crop_image(const unsigned char *src, int src_wdith, int src_height, int src_channel, int width_step,
                const cv::Rect &roi, unsigned char *dst_buffer);

bool gray_image(const void *src, int src_wdith, int src_height, int src_channels, void *dst_buf);

bool repeat_gray_image(const void *src, int src_wdith, int src_height, int dst_channels, void *dst_buf);