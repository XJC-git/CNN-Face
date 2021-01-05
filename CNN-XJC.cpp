#define File_Path "C:\\Files\\Samples\\face05.jpg"
#include <iostream>
#include<opencv2/opencv.hpp>
#include<thread>
#include<cmath>
#include "param.h"
using namespace cv;
float** transMat(uchar* image);
float** ConvBNRelu(float** image, conv_param param, int in_size);
void MatCopy(float* a, float* b, int copy_size, int in_size, int index);
float** MaxPool(float** input, int channel, int size);
float* FullConnect(float** input, int channel, fc_param param);
void DotProduct_thread(float* a, float* b, int size, float* ans,int index);
int main()
{
    Mat image_BGR = imread(File_Path);
    uchar* image = image_BGR.data;
    std::cout << image[2]/255.0f<<std::endl;
    float** image_RGB = transMat(image);
    float** trans;
    printf("完成准备....\n");
    trans = ConvBNRelu(image_RGB, conv_params[0], 128);
    trans = MaxPool(trans, 16, 64);
    printf("第一层\n");
    trans = ConvBNRelu(trans, conv_params[1], 32);
    trans = MaxPool(trans, 32, 32);
    trans = ConvBNRelu(trans, conv_params[2], 16);
    float* answer = FullConnect(trans, 32, fc_params[0]);
    float sum = exp(answer[0]) + exp(answer[1]);
    float a1 = exp(answer[0]) / sum;
    float a2 = exp(answer[1]) / sum;
    printf("背景分数：%.8f\n", a1);
    printf("人脸分数：%.8f\n", a2);
}
float** transMat(uchar* image) {
    float** ans = new float * [3];
    int size = 128 * 128;
    float* R = new float[size];
    float* G = new float[size];
    float* B = new float[size];
    int count = 0;
    for (int i = 0; i < size; i++) {
        B[i] = image[count++]/255.0f;
        G[i] = image[count++]/255.0f;
        R[i] = image[count++]/255.0f;
    }
    ans[0] = R;
    ans[1] = G;
    ans[2] = B;
    return ans;
}
float** ConvBNRelu(float** image, conv_param param,int in_size) {
    int pad = param.pad;
    int stride = param.stride;
    int kernel_size=param.kernel_size;
    int in_channels=param.in_channels;
    int out_channels=param.out_channels;
    int out_size = ((in_size + 2 * pad) - kernel_size + 1) / stride;
    int size_channel = out_channels * in_channels;
    
    float** channel = new float* [size_channel];
    int size_k = kernel_size * kernel_size;
    int size_m = out_size * out_size;
    int count = 0;
    for (int i = 0; i < size_channel; i++) {
        float* temp = new float[size_k];
        for (int j = 0; j < size_k; j++) {
            temp[j] = param.p_weight[count++];
        }
        channel[i] = temp;
    }
    float** ans = new float* [out_channels];
    //填充矩阵
    if (pad) {
        
        int new_size_s = (in_size + 2 * pad);
        int new_size = new_size_s * new_size_s;
        for (int i = 0; i < in_channels; i++) {
            float* origin = image[i];
            float* after = new float[new_size];
            int count_a = 0;
            for (int j = 0; j < new_size_s; j++) {
                int pos = (j - 1) * in_size;
                for (int k = 0; k < new_size_s; k++) {
                    if (j == 0 || k == 0||j==new_size_s-1||k==new_size_s-1) {
                        after[count_a++] = 0;
                    }
                    else {
                        after[count_a++] = origin[pos++];
                    }
                }
            }
            image[i] = after;
            delete[] origin;
        }
        in_size += 2;
    }
    printf("填充完成\n");
    
    //分割矩阵
    std::thread* threadPool = new std::thread[out_size];
    float*** mat_cut = new float** [in_channels];
    for (int m = 0; m < in_channels; m++) {
        float** temp_channel = new float* [size_m];
        int count = 0;
        for (int i = 0; i < out_size; i++) {
            int index = (stride * i) * in_size;
            for (int j = 0; j < out_size; j++) {
                float* temp = new float[size_k];
                MatCopy(image[m], temp,kernel_size, in_size, index);
                temp_channel[count++] = temp;
                index += stride;
            }
            for (int j = 0; j < out_size; j++) {
                //threadPool[j].join();//启动线程
            }
        }
        mat_cut[m] = temp_channel;
    }
    printf("分割完成\n");
    threadPool = new std::thread[in_channels];
    int count_c = 0;
    for (int i = 0; i < out_channels; i++) {//产生通道数对应的矩阵
        float* temp = new float[size_m];
        int count_b = 0;
        for (int j = 0; j < out_size; j++) {
            for (int k = 0; k < out_size; k++) {
                float* temp_num = new float[in_channels] {0};
                for (int m = 0; m < in_channels; m++) {//开始卷积
                    DotProduct_thread(mat_cut[m][count_b], channel[count_c++], size_k,temp_num,m);
                }
                count_c -= in_channels;
                for (int m = 0; m < in_channels; m++) {
                    //threadPool[m].join();//启动线程
                }
                float trans = 0;
                for (int m = 0; m < in_channels; m++) {
                    trans += temp_num[m];
                }
                trans += param.p_bias[i];
                temp[count_b] = trans<0?0:trans;
                count_b++;
            }
        }
        ans[i] = temp;
        count_c += in_channels;
    }
    delete[] channel;
    return ans;
}
float** MaxPool(float** input, int channel,int size) {
    float** ans = new float* [channel];
    int mat_size = (size / 2) * (size / 2);
    for (int i = 0; i < channel; i++) {
        float* temp = new float[mat_size];
        int count1 = 0; int count2 = size; int count = 0;
        for (int j = 0; j < size; j += 2) {
            for (int k = 0; k < size; k += 2) {
                float a = input[i][count1++];
                float b = input[i][count1++];
                float c = input[i][count2++];
                float d = input[i][count2++];
                if (a < b)
                    a = b;
                if (b< c)
                    a = c;
                if (c < d)
                    a = d;
                temp[count++] = a;
            }
            count1 += size;
            count2 += size;
        }
        ans[i] = temp;
    }
    return ans;
}
float* FullConnect(float** input, int channel, fc_param param) {
    float* ans = new float[2]{ 0 };
    int count = 0; int count_m = 0;
    for (int i = 0; i < channel; i++) {
        while (count_m < 64) {
            ans[0] += (param.p_weight[count++] * input[i][count_m++]);
        }
        count_m = 0;
    }
    ans[0] += param.p_bias[0];
    for (int i = 0; i < channel; i++) {
        while (count_m < 64) {
            ans[1] += (param.p_weight[count++] * input[i][count_m++]);
        }
        count_m = 0;
    }
    ans[1] += param.p_bias[1];
    return ans;
}
void DotProduct_thread(float* a,float* b,int size,float* ans,int index) {
    for (int i = 0; i < size; i++) {
        ans[index] += a[i] * b[i];
    }
}
void MatCopy(float* a, float* b, int copy_size, int in_size,int index) {
    int count = 0; 
    for (int i = 0; i < copy_size;i++) {
        int pl = index + i * in_size;
        for (int j = 0; j < copy_size; j++) {
            b[count++] = a[pl++];
        }
    }
}

