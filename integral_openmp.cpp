#include <iostream>
#include <iomanip>
#include <time.h>
#include <math.h>

#define DO_PRINT false

double func(auto x);

double calc_integral(auto seg_count, auto a, auto b);
double calc_runge_error(auto sum_prev_step, auto sum_cur_step);

void print_step(auto iter, auto seg_count, auto sum, auto err_runge);
void print_step_winner(auto iter, auto seg_count, auto sum, auto err_runge);
void print_header();
void print_footer();

double calc_integral_with_accuracy(auto a, auto b, auto accuracy, auto init_segments_count);

int main()
{
    auto a = 1.0;
    auto b = 2.0;
    auto accuracy = 1E-10;
    auto init_segments_count = 6;

    auto start = clock();
    auto result = calc_integral_with_accuracy(a, b, accuracy, init_segments_count);
    auto end = clock();
    std::cout << std::fixed;
    std::cout << std::setprecision(12);
    std::cout << "result: " << result << std::endl;
    std::cout << "time: " << (double)(end - start) / CLOCKS_PER_SEC << " sec" << std::endl;
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
    print_header();

    auto err_runge = 1.0;
    auto seg_count = init_segments_count;
    auto iter = 1;

    auto sum_prev_step = calc_integral(seg_count, a, b);
    seg_count += 2;

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
