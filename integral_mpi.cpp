#include "mpi.h"

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
    MPI_Init(NULL, NULL);

    int world_rank;
    int world_size;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_rank <= 2)
    {
        auto a = 1.0;
        auto b = 2.0;
        auto accuracy = 1E-50;
        auto init_segments_count = 10;

        auto start = std::chrono::high_resolution_clock::now();
        auto result = calc_integral_with_accuracy(a, b, accuracy, init_segments_count);
        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds"
                  << ", process: " << world_rank << std::endl;
        if (world_rank == 0)
        {
            std::cout << "result: " << std::fixed << std::setprecision(30) << result << std::endl;
        }
    }

    MPI_Finalize();
}

double func(auto x)
{
    return 2 - x + cos(x) - log10(1 + x);
}

double calc_runge_error(auto sum_prev_step, auto sum_cur_step)
{
    return abs(sum_cur_step - sum_prev_step);
}

double calc_integral(auto seg_count, auto a, auto b)
{
    auto step_size = (b - a) / seg_count;
    auto sum = 0.0;

    for (auto i = 0; i < seg_count; i++)
    {
        auto x_left = a + i * step_size;
        auto x_right = a + (i + 1) * step_size;
        sum += ((x_right - x_left) / 6) * (func(x_left) + 4 * func((x_left + x_right) / 2) + func(x_right));
    }

    return sum;
}

double calc_integral_with_accuracy(auto a, auto b, auto accuracy, auto init_segments_count)
{
    int world_rank;
    int world_size;

    // vars for 0 process
    int seg_count = init_segments_count;
    double err_runge;
    int iter;
    double sum_from_1;
    double sum_from_2;

    // vars for 1 and 2 processes
    double sum_on_proc;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // send seg_count to 1 and 2 processes
    if (world_rank == 0)
    {
        print_header();
        auto seg_count_for_1_process = init_segments_count;
        auto seg_count_for_2_process = init_segments_count + 2;
        MPI_Send(&seg_count_for_1_process, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Send(&seg_count_for_2_process, 1, MPI_INT, 2, 0, MPI_COMM_WORLD);

        std::cout << "Proc 0 send seg_count: " << seg_count_for_1_process << " to Proc 1" << std::endl;
        std::cout << "Proc 0 send seg_count: " << seg_count_for_2_process << " to Proc 2" << std::endl;

        err_runge = 1.0;
        iter = 1;
    }
    else
    {
        // receive seg_count from 0 process
        MPI_Recv(&seg_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Proc " << world_rank << " receive seg_count: " << seg_count << " from Proc 0" << std::endl;
        sum_on_proc = calc_integral(seg_count, a, b);
        print_step(iter, seg_count, sum_on_proc, err_runge);
    }

    // get sum from 1 and 2 processes
    if (world_rank == 0)
    {
        MPI_Recv(&sum_from_1, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&sum_from_2, 1, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        err_runge = calc_runge_error(sum_from_1, sum_from_2);
        MPI_Send(&err_runge, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        MPI_Send(&err_runge, 1, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Send(&sum_on_proc, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Recv(&err_runge, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (world_rank == 0)
    {
        auto sum_prev_step = sum_from_2;

        while (err_runge > accuracy)
        {
            print_step(iter, seg_count, sum_prev_step, err_runge);
            auto sum_cur_step = calc_integral(seg_count, a, b);
            seg_count += 2;
            err_runge = calc_runge_error(sum_prev_step, sum_cur_step);
            sum_prev_step = sum_cur_step;
            iter++;
        }

        print_step_winner(iter, seg_count, sum_prev_step, err_runge);
        print_footer();

        return sum_prev_step;
    }

    return 0.0;
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
