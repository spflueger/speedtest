#include <algorithm>
#include <vector>
#include <pstl/algorithm>
#include <pstl/execution>
#include <chrono>
#include <iostream>
#include <random>

int main()
{
    size_t vecsize(1000000);
    std::vector<double> a(vecsize);
    std::vector<double> b(vecsize);

    std::vector<double> tempres1(vecsize);
    std::vector<double> tempres2(vecsize);
    std::vector<double> tempres3(vecsize);

    std::vector<double> result(vecsize);

    auto seed = std::chrono::system_clock::now().time_since_epoch().count(); //seed
    std::default_random_engine dre(seed);                                    //engine
    std::uniform_real_distribution<double> di(-100000.0, 100000.0);          //distribution

    std::generate(a.begin(), a.end(), [&] { return di(dre); });
    std::generate(b.begin(), b.end(), [&] { return di(dre); });

    typedef std::chrono::duration<long double> MySecondTick;
    MySecondTick sec(0);
    size_t loops(100);
    for (size_t i = 0; i < loops; ++i)
    {
        std::chrono::steady_clock::time_point StartTime =
            std::chrono::steady_clock::now();

        // directly calculate everything...
        // std::transform(pstl::execution::par_unseq, a.begin(), a.end(), b.begin(), result.begin(), [](double x, double y) { return std::sqrt(std::abs(std::cos(x) * y)); });

        // do each part separately
        std::transform(pstl::execution::par_unseq, a.begin(), a.end(), tempres1.begin(), [](double x) { return std::cos(x); });
        std::transform(pstl::execution::par_unseq, tempres1.begin(), tempres1.end(), b.begin(), tempres2.begin(), [](double x, double y) { return x * y; });
        std::transform(pstl::execution::par_unseq, tempres2.begin(), tempres2.end(), tempres3.begin(), [](double x) { return std::abs(x); });
        std::transform(pstl::execution::par_unseq, tempres3.begin(), tempres3.end(), result.begin(), [](double x) { return std::sqrt(x); });

        std::chrono::steady_clock::time_point EndTime =
            std::chrono::steady_clock::now();

        sec += (EndTime - StartTime);

        for (size_t i = 0; i < vecsize; ++i)
            assert(std::abs(result[i] - std::sqrt(std::abs(std::cos(a[i]) * b[i]))) < 1e-10);
    }
    sec /= loops;
    std::cout << sec.count() << " seconds " << std::endl;
}