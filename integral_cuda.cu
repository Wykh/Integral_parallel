#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <iomanip>
#include <time.h>
#include <math.h>
#include <chrono>
#include <unistd.h>

#define DO_PRINT false

double func(auto x);

double calc_integral(auto seg_count, auto a, auto b);
double calc_integral_with_accuracy(auto a, auto b, auto accuracy, auto init_segments_count);
double calc_runge_error(auto sum_prev_step, auto sum_cur_step);

void print_step(auto iter, auto seg_count, auto sum, auto err_runge);
void print_step_winner(auto iter, auto seg_count, auto sum, auto err_runge);
void print_header();
void print_footer();

int main()
{
    auto a = 1.0;
    auto b = 2.0;
    auto accuracy = 1E-50;
    auto init_segments_count = 100000;

    auto start = std::chrono::high_resolution_clock::now();
    auto result = calc_integral_with_accuracy(a, b, accuracy, init_segments_count);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "result: " << std::fixed << std::setprecision(30) << result << std::endl;
    std::cout << "time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds" << std::endl;
}

double calc_runge_error(auto sum_prev_step, auto sum_cur_step)
{
    return abs(sum_cur_step - sum_prev_step);
}

//device func
__device__ double func_device(double x)
{
    return 2 - x + cos(x) - log10(1 + x);
}

// cuda kernal calc_integral
__global__ void calc_integral_kernel(double *a_d, double *b_d, double *step_size_d, double *sum_d, int seg_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < seg_count)
    {
        double x_left = a_d[0] + i * step_size_d[0];
        double x_right = a_d[0] + (i + 1) * step_size_d[0];
        sum_d[i] = ((x_right - x_left) / 6) * (func_device(x_left) + 4 * func_device((x_left + x_right) / 2) + func_device(x_right));
    }
}

double calc_integral_with_accuracy(auto a, auto b, auto accuracy, auto init_segments_count)
{
    print_header();

    auto err_runge = 1.0;
    auto seg_count = init_segments_count;
    auto iter = 1;

    // allocate memory on host
    double *a_h = (double *)malloc(sizeof(double));
    double *b_h = (double *)malloc(sizeof(double));
    double *step_size_h = (double *)malloc(sizeof(double));
    double *sum_h = (double *)malloc(sizeof(double) * seg_count);

    // allocate memory on device
    double *a_d;
    double *b_d;
    double *step_size_d;
    double *sum_d;
    cudaMalloc((void **)&a_d, sizeof(double));
    cudaMalloc((void **)&b_d, sizeof(double));
    cudaMalloc((void **)&step_size_d, sizeof(double));
    cudaMalloc((void **)&sum_d, sizeof(double) * seg_count);

    // copy data from host to device
    a_h[0] = a;
    b_h[0] = b;
    cudaMemcpy(a_d, a_h, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, sizeof(double), cudaMemcpyHostToDevice);

    // calculate step size
    step_size_h[0] = (b - a) / seg_count;
    cudaMemcpy(step_size_d, step_size_h, sizeof(double), cudaMemcpyHostToDevice);

    // calculate sum
    calc_integral_kernel<<<(seg_count + 255) / 256, 256>>>(a_d, b_d, step_size_d, sum_d, seg_count);
    cudaMemcpy(sum_h, sum_d, sizeof(double) * seg_count, cudaMemcpyDeviceToHost);

    // sum up the result
    auto sum_prev_step = 0.0;
    for (auto i = 0; i < seg_count; i++)
    {
        sum_prev_step += sum_h[i];
    }

    seg_count += 2;

    while (err_runge > accuracy)
    {
        print_step(iter, seg_count, sum_prev_step, err_runge);

        // calculate step size
        step_size_h[0] = (b - a) / seg_count;
        cudaMemcpy(step_size_d, step_size_h, sizeof(double), cudaMemcpyHostToDevice);

        // calculate sum
        calc_integral_kernel<<<(seg_count + 255) / 256, 256>>>(a_d, b_d, step_size_d, sum_d, seg_count);
        cudaMemcpy(sum_h, sum_d, sizeof(double) * seg_count, cudaMemcpyDeviceToHost);

        // sum up the result
        auto sum_cur_step = 0.0;
        for (auto i = 0; i < seg_count; i++)
        {
            sum_cur_step += sum_h[i];
        }

        seg_count += 2;
        err_runge = calc_runge_error(sum_prev_step, sum_cur_step);
        sum_prev_step = sum_cur_step;
        iter++;
    }

    print_step_winner(iter, seg_count, sum_prev_step, err_runge);
    print_footer();

    // free memory on device
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(step_size_d);
    cudaFree(sum_d);

    // free memory on host
    free(a_h);
    free(b_h);
    free(step_size_h);
    free(sum_h);

    return sum_prev_step;
}

void print_step(auto iter, auto seg_count, auto sum, auto err_runge)
{
    if (!DO_PRINT)
        return;
    std::cout << std::setw(5) << iter << " | ";
    std::cout << std::setw(9) << seg_count << " | ";
    std::cout << std::setw(15) << std::setprecision(15) << sum << " | ";
    std::cout << std::setw(8) << std::setprecision(3) << err_runge << std::endl;
}

void print_step_winner(auto iter, auto seg_count, auto sum, auto err_runge)
{
    if (!DO_PRINT)
        return;
    std::cout << std::setw(5) << iter << " | ";
    std::cout << std::setw(9) << seg_count << " | ";
    std::cout << std::setw(15) << std::setprecision(15) << sum << " | ";
    std::cout << std::setw(8) << std::setprecision(3) << err_runge;
    std::cout << " <--- winner" << std::endl;
}

void print_header()
{
    if (!DO_PRINT)
        return;
    std::cout << std::setw(5) << "iter"
              << " | ";
    std::cout << std::setw(9) << "seg_count"
              << " | ";
    std::cout << std::setw(17) << "sum"
              << " | ";
    std::cout << std::setw(8) << "err_runge" << std::endl;

    std::cout << std::setw(5) << "----"
              << " | ";
    std::cout << std::setw(9) << "---------"
              << " | ";
    std::cout << std::setw(17) << "-----------------"
              << " | ";
    std::cout << std::setw(8) << "---------" << std::endl;
}

void print_footer()
{
    if (!DO_PRINT)
        return;
    std::cout << std::setw(5) << "----"
              << " | ";
    std::cout << std::setw(9) << "---------"
              << " | ";
    std::cout << std::setw(17) << "-----------------"
              << " | ";
    std::cout << std::setw(8) << "---------" << std::endl;
}
