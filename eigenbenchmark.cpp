#include <Eigen/Dense>
#include <vector>
#include <chrono>
#include <iostream>
#include <random>

int main()
{
    // the mt does not work for these operations...
    Eigen::setNbThreads(4);
    size_t vecsize(1000000);
    Eigen::ArrayXd a(vecsize);
    Eigen::ArrayXd b(vecsize);

    auto seed = std::chrono::system_clock::now().time_since_epoch().count(); //seed
    std::default_random_engine dre(seed);                                    //engine
    std::uniform_real_distribution<double> di(-100000.0, 100000.0);          //distribution

    for (size_t i = 0; i < vecsize; ++i)
    {
        a(i) = di(dre);
        b(i) = di(dre);
    }

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
        auto result = ((a.cos() * b).abs().sqrt()).eval();
        assert(std::abs(result[0] - std::sqrt(std::abs(std::cos(a[0]) * b[0]))) < 1e-10);
        assert(std::abs(result[vecsize - 1] - std::sqrt(std::abs(std::cos(a[vecsize - 1]) * b[vecsize - 1]))) < 1e-10);
        std::chrono::steady_clock::time_point EndTime =
            std::chrono::steady_clock::now();

        sec += (EndTime - StartTime);

        for (size_t i = 0; i < vecsize; ++i)
            assert(std::abs(result[i] - std::sqrt(std::abs(std::cos(a[i]) * b[i]))) < 1e-10);
    }
    sec /= loops;
    std::cout << sec.count() << " seconds " << std::endl;
}