#include "process.h"

#include "npp_common.h"

bool resize_image(const void *src, int src_wdith, int src_height, int width_step, void *dst_buf, int dst_width,
                  int dst_height)
{
    if (!src || !dst_buf)
    {
        return false;
    }
    NppiSize srcsize = {src_wdith, src_height};
    NppiRect srcroi = {0, 0, src_wdith, src_height};
    NppiSize dstsize = {dst_width, dst_height};
    NppiRect dstroi = {0, 0, dst_width, dst_height};

    NPP_CHECK(nppiResize_8u_C3R((Npp8u *)src, width_step, srcsize, srcroi, (Npp8u *)dst_buf, dst_width * 3, dstsize,
                                dstroi, NPPI_INTER_LINEAR));
    return true;
}

bool gray_image(const void *src, int src_wdith, int src_height, int src_channels, void *dst_buf)
{
    if (!src)
    {
        return false;
    }
    Npp32f aCoeffs[3] = {0.114, 0.587, 0.299};
    NppiSize oSrcSizeROI = {src_wdith, src_height};
    NPP_CHECK(nppiColorToGray_8u_C3C1R((Npp8u *)src, src_wdith * src_channels, (Npp8u *)dst_buf, src_wdith, oSrcSizeROI,
                                       aCoeffs));
    // nppiRGBToGray_8u_C3C1R((Npp8u *)src, width_step, (Npp8u *)dst_buf, width_step, oSrcSizeROI);
    return true;
}

bool repeat_gray_image(const void *src, int src_wdith, int src_height, int dst_channels, void *dst_buf)
{
    if (!src)
    {
        return false;
    }
    NppiSize oDstSizeROI = {src_wdith, src_height};
    NPP_CHECK(nppiDup_8u_C1C3R((Npp8u *)src, src_wdith, (Npp8u *)dst_buf, src_wdith * dst_channels, oDstSizeROI));
    return true;
}

bool padding_image(const void *src, int src_wdith, int src_height, int width_step, unsigned char pad_vaule[3],
                   void *dst_buf, int dst_wid, int dst_hei)
{
    if (!src || !dst_buf)
    {
        return false;
    }
    Npp8u aValue[3] = {pad_vaule[0], pad_vaule[1], pad_vaule[2]};
    int padding_pos_x = std::max((src_wdith - dst_wid) / 2, 0);
    int padding_pos_y = std::max((src_height - dst_hei) / 2, 0);
    NppiSize oDstSizeROI = {dst_wid, dst_hei};
    NppiSize oSrcSizeROI = {src_wdith, src_height};
    NPP_CHECK(nppiCopyConstBorder_8u_C3R((Npp8u *)src, width_step, oSrcSizeROI, (Npp8u *)dst_buf, dst_wid * 3,
                                         oDstSizeROI, padding_pos_y, padding_pos_x, aValue));
    return true;
}

bool normlize_image(float *src, int src_wdith, int src_height, int channel, float mean[3], float std_val[3])
{
    if (!src)
    {
        return false;
    }

    NppiSize dstsize = {src_wdith, src_height};
    NPP_CHECK(nppiSubC_32f_C3IR(mean, (Npp32f *)src, src_wdith * channel * sizeof(float), dstsize));

    NPP_CHECK(nppiDivC_32f_C3IR(std_val, (Npp32f *)src, src_wdith * channel * sizeof(float), dstsize));
    return true;
}

bool crop_image(const unsigned char *src, int src_wdith, int src_height, int src_channel, int width_step,
                const cv::Rect &roi, unsigned char *dst_buffer)
{
    if (!src || !dst_buffer)
    {
        return false;
    }
    int offset = roi.y * width_step + roi.x * src_channel;
    // int dst_buf_bytes = roi.width * roi.height * src_channel;
    // if (dst_buf_bytes > dst_buffer.size()) {
    //   dst_buffer.resize(dst_buf_bytes);
    // }

    NppiSize dst_size = {roi.width, roi.height};
    NPP_CHECK(
        nppiCopy_8u_C3R((Npp8u *)(src + offset), width_step, (Npp8u *)dst_buffer, roi.width * src_channel, dst_size));
    return true;
}