#include <functional>
#include <algorithm>
#include <vector>
#include <pstl/algorithm>
#include <pstl/execution>
#include <chrono>
#include <iostream>
#include <random>
#include <any>
#include <complex>
#include <memory>

// define sinks and sources
// for this local graph sinks and sources are just vectors of an arbitrary type
// the node operator will take the sources, operate on them/transform them to the sink
// our node operators will only have a single sink, but the sink can be a source to
// multiple nodes above...
// (so here we already see that a sink and a source are basically just these vectors)
// the lifetime of these sinks and sources just have to be managed accordingly
// I think a source that is used multiple times should be always wrapped by a caching node
// so that this information will only be calculated once
// also once the tree is finished, all branches which cannot change due to fit parameter changes
// should be cached at that point anyways. these points would also be the starting point of a branch pipeline
// what we mention below
/* template <typename T>
struct Sink
{
    std::unique_ptr<T> data;
}

template <typename T>
struct Source
{
    std::unique_ptr<T> data;
}

template <typename Input, typename Output>
class UnaryPipelineProcess
{
    process();
    Sink<Output> sink;
    Source<Input> source;
}*/

class OperationStrategy
{
public:
    virtual ~OperationStrategy() = default;

    virtual void execute() = 0;
};

// defines a standard binary operation in a vectorized way using std::vector
template <typename BinaryOperator, typename InputType1, typename InputType2, typename OutputType>
class VectorizedBinaryOperationFunctor : public OperationStrategy
{
public:
    VectorizedBinaryOperationFunctor(
        auto function, auto const &input1, auto const &input2, auto &output)
        : Input1(input1), Input2(input2), Output(output),
          Function(function) {}

    void execute() final
    {
        std::transform(pstl::execution::par_unseq, Input1.begin(), Input1.end(), Input2.begin(),
                       Output.begin(), Function);
    }

private:
    const std::vector<InputType1> &Input1;
    const std::vector<InputType2> &Input2;
    std::vector<OutputType> &Output;
    BinaryOperator Function;
};

// defines a standard binary operation in a vectorized way using std::vector
template <typename UnaryOperator, typename InputType, typename OutputType>
class VectorizedUnaryOperationFunctor : public OperationStrategy
{
public:
    VectorizedUnaryOperationFunctor(
        UnaryOperator function, const std::vector<OutputType> &input, std::vector<OutputType> &output)
        : Input(input), Output(output),
          Function(function) {}

    void execute() final
    {
        std::transform(pstl::execution::par_unseq, Input.begin(), Input.end(),
                       Output.begin(), Function);
    }

private:
    const std::vector<InputType> &Input;
    std::vector<OutputType> &Output;
    UnaryOperator Function;
};

template <typename T>
class StrategyCachingDecorator : public OperationStrategy
{
public:
    StrategyCachingDecorator() {}
    void execute() final
    {
        if (false)
        {
            UndecoratedStrategy->execute();
        }
    }

private:
    std::unique_ptr<OperationStrategy> UndecoratedStrategy;
    std::vector<T> CachedResult;
};

template <typename T>
struct Value
{
    std::string Name;
    unsigned int UniqueID;
    T Value;
};

using DataPoint = std::vector<double>;
using ParameterList = std::vector<std::any>;

template <typename OutputType>
class Function
{
public:
    virtual ~Function() = default;
    virtual OutputType evaluate() const = 0;
    // changes parameters to the given values in the list
    virtual void updateParametersFrom(const ParameterList &list) = 0;
    // gets a list of parameters defined by this function
    virtual ParameterList getParameters() const = 0;
};

using Intensity = Function<std::vector<double>>;
using Amplitude = Function<std::vector<std::complex<double>>>;

// another thing is how is the whole graph going to be executed.
// for example: we can have a set of pipelines (a branch pipeline, which would be just a vector of connected operations)
// depending on the parameters which changed. then at each fit iteration, we just execute the pipeline that corresponds
// to that change and run that pipeline. the pipeline itself does not take much storage in memory, so having multiple of the does not matter
// then res should be just the vector of the results that we want to return from the interface

using DataID = size_t;

template <typename OutputType>
class VectorizedFunctionGraph : public Function<OutputType>
{
public:
    VectorizedFunctionGraph() = default;
    virtual ~VectorizedFunctionGraph() = default;

    template <typename Function, typename Output, typename Input1, typename Input2>
    DataID
    addBinaryNode(Function f, DataID a, DataID b)
    {
        std::vector<Output> out(std::any_cast<const std::vector<Input1> &>(Storage[a]).size());
        Storage.push_back(out);
        Nodes.push_back(std::unique_ptr<OperationStrategy>(
            new VectorizedBinaryOperationFunctor<Function, Input1, Input2, Output>(
                f, std::any_cast<const std::vector<Input1> &>(Storage.at(a)),
                std::any_cast<const std::vector<Input2> &>(Storage.at(b)),
                std::any_cast<std::vector<Output> &>(Storage.back()))));
        return Storage.size() - 1;
    }

    template <typename T>
    DataID createDataSource(T data)
    {
        Storage.push_back(data);
        return Storage.size() - 1;
    }

    OutputType evaluate() const
    {
        for (const std::unique_ptr<OperationStrategy> &node : Nodes)
        {
            node->execute();
        }
        return std::any_cast<OutputType>(Storage.back());
    }

    void updateParametersFrom(const ParameterList &list)
    {
    }

    ParameterList getParameters() const
    {
    }

    void createPipelines()
    {
    }

private:
    //TODO: use maps here?
    std::vector<std::unique_ptr<OperationStrategy>> Nodes;
    std::vector<std::any> Storage;
};

int main()
{
    size_t vecsize(1000000);
    std::vector<double> a(vecsize);
    std::vector<double> b(vecsize);

    auto seed = std::chrono::system_clock::now().time_since_epoch().count(); //seed
    std::default_random_engine dre(seed);                                    //engine
    std::uniform_real_distribution<double> di(-100000.0, 100000.0);          //distribution

    std::generate(a.begin(), a.end(), [&] { return di(dre); });
    std::generate(b.begin(), b.end(), [&] { return di(dre); });

    typedef std::chrono::duration<long double> MySecondTick;
    MySecondTick sec(0);
    size_t loops(100);

    VectorizedFunctionGraph<std::vector<double>> g;

    auto ida = g.createDataSource<std::vector<double>>(a);
    auto idb = g.createDataSource<std::vector<double>>(b);

    auto resid = g.addBinaryNode<std::multiplies<double>, double, double, double>(std::multiplies<double>(), ida, idb);

    //auto mycos_wrapper = [](double x) { return std::cos(x); };
    //auto myabs_wrapper = [](double x) { return std::abs(x); };
    //auto mysqrt_wrapper = [](double x) { return std::sqrt(x); };
    //auto node1 = VectorizedUnaryOperationFunctor<decltype(mycos_wrapper), double, double>(mycos_wrapper, a, tempres1);
    //auto node2 = VectorizedBinaryOperationFunctor<std::multiplies<double>, double, double, double>(std::multiplies<double>(), tempres1, b, tempres2);
    //auto node3 = VectorizedUnaryOperationFunctor<decltype(myabs_wrapper), double, double>(myabs_wrapper, tempres2, tempres3);
    //auto node4 = VectorizedUnaryOperationFunctor<decltype(mysqrt_wrapper), double, double>(mysqrt_wrapper, tempres3, result);

    for (size_t i = 0; i < loops; ++i)
    {
        std::chrono::steady_clock::time_point StartTime =
            std::chrono::steady_clock::now();

        auto result = g.evaluate();

        std::chrono::steady_clock::time_point EndTime =
            std::chrono::steady_clock::now();

        sec += (EndTime - StartTime);

        for (size_t i = 0; i < vecsize; ++i)
            assert(std::abs((a[i] * b[i]) - result[i]) < 1e-10);
            //assert(std::abs(result[i] - std::sqrt(std::abs(std::cos(a[i]) * b[i]))) < 1e-10);
    }
    sec /= loops;
    std::cout << sec.count() << " seconds " << std::endl;
}