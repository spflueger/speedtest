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
#include <map>

// defines fundamental data types: double, complex, matrix
//enum class DataType = {INT, DOUBLE, COMPLEX_DOUBLE}

struct Dimension
{
    size_t Size;
};

struct Tensor
{
    virtual ~Tensor() = default;
    std::vector<Dimension> Dimensions;
    // size: scalar = 0, vector = 1, matrix = 2
};

template <typename T>
struct Scalar : public Tensor
{
    Scalar(T value) : Value(value) {}
    T Value;
};

template <typename T>
struct Vector : public Tensor
{
    Vector() = default;
    Vector(std::vector<T> values) : Values(values)
    {
        Dimensions.push_back({Values.size()});
    }
    std::vector<T> Values;
};

// matrix
template <typename T>
struct Matrix : public Tensor
{
    std::vector<std::vector<T>> Values;
};

template <typename T>
struct Value
{
    std::string Name;
    T Value;
};

using DataPoint = std::vector<double>;
using ParameterList = std::vector<std::pair<unsigned int, double>>;

template <typename OutputType, typename... InputTypes>
class Function
{
public:
    virtual ~Function() = default;
    // we dont make the evaluate function const anymore, because
    // it will allow internal modification like caching
    virtual OutputType evaluate(const InputTypes &... args) = 0;
    // changes parameters to the given values in the list
    // The order of the parameters in the list is important.
    // It has to be the same as returned by getParameters()
    virtual void updateParametersFrom(const ParameterList &) = 0;
    // gets a list of parameters defined by this function
    virtual ParameterList getParameters() const = 0;
};

// and intensity is just a function which takes a list of data points and returns a list of intensities
using Intensity = Function<std::vector<double>, std::vector<DataPoint>>;

// Define estimators as a function with no input
template <typename OutputType>
class Estimator : public Function<OutputType>
{
};

// this is what a log likelihood estimator would look like
class MaxLogLHEstimator : public Estimator<double>
{
    double evaluate() final;
    // changes parameters to the given values in the list
    void updateParametersFrom(const ParameterList &) final;
    // gets a list of parameters defined by this function
    ParameterList getParameters() const final;
};

struct OptimizationSettings
{
    std::map<std::string, bool> FixedParameters;
};

class Optimizer
{
    // Get the list of parameters from the estimator (via getParameters())
    // Compare with the Settings, and create a mapping of the fitted parameters
    // to their place in the complete list
    void optimize(const Estimator<double> &Estimator, OptimizationSettings Settings);
};

// ------------------------ Now we define your ComPWA function graph backend! -------------------------

class OperationStrategy
{
public:
    virtual ~OperationStrategy() = default;

    virtual void execute() = 0;
};

// TODO: use result type traits to predefine the output type given the input types!

// defines a standard binary operation in a vectorized way using std::vector
template <typename OutputType, typename InputType1, typename InputType2, typename BinaryOperator>
class BinaryOperationFunctor : public OperationStrategy
{
public:
    BinaryOperationFunctor(
        std::shared_ptr<OutputType> output, std::shared_ptr<const InputType1> input1, std::shared_ptr<const InputType2> input2, BinaryOperator function)
        : Input1(input1), Input2(input2), Output(output),
          Function(function) {}

    void execute() final
    {
        Function(*Output, *Input1, *Input2);
    }

private:
    std::shared_ptr<const InputType1> Input1;
    std::shared_ptr<const InputType2> Input2;
    std::shared_ptr<OutputType> Output;
    BinaryOperator Function;
};

template <typename BinaryFunction>
struct ElementWiseBinaryOperation
{
    ElementWiseBinaryOperation(BinaryFunction f) : Function(f){};

    template <typename OutputType, typename InputType1, typename InputType2>
    void operator()(Scalar<OutputType> &Output, const Scalar<InputType1> &Input1, const Scalar<InputType2> &Input2)
    {
        Output.Value = Function(Input1.Value, Input2.Value);
    }

    template <typename OutputType, typename InputType1, typename InputType2>
    void operator()(Vector<OutputType> &Output, const Vector<InputType1> &Input1, const Vector<InputType2> &Input2)
    {
        std::transform(pstl::execution::par_unseq, Input1.Values.begin(), Input1.Values.end(), Input2.Values.begin(),
                       Output.Values.begin(), Function);
    }

    template <typename OutputType, typename InputType1, typename InputType2>
    void operator()(Vector<OutputType> &Output, const Vector<InputType1> &Input1, const Scalar<InputType2> &Input2)
    {
        std::transform(pstl::execution::par_unseq, Input1.Values.begin(), Input1.Values.end(),
                       Output.Values.begin(), [&Input2, this](const InputType1 &x) { return Function(x, Input2.Value); });
    }

private:
    BinaryFunction Function;
};

// defines a standard unary operation in a vectorized way using std::vector
template <typename OutputType, typename InputType, typename UnaryOperator>
class UnaryOperationFunctor : public OperationStrategy
{
public:
    UnaryOperationFunctor(
        std::shared_ptr<OutputType> output, std::shared_ptr<const InputType> input, UnaryOperator function)
        : Input(input), Output(output),
          Function(function) {}

    void execute() final
    {
        Function(*Output, *Input);
    }

private:
    std::shared_ptr<const InputType> Input;
    std::shared_ptr<OutputType> Output;
    UnaryOperator Function;
};

template <typename UnaryFunction>
struct ElementWiseUnaryOperation
{
    ElementWiseUnaryOperation(UnaryFunction f) : Function(f){};

    template <typename OutputType, typename InputType>
    void operator()(Scalar<OutputType> &Output, const Scalar<InputType> &Input)
    {
        Output.Value = Function(Input.Value);
    }

    template <typename OutputType, typename InputType>
    void operator()(Vector<OutputType> &Output, const Vector<InputType> &Input)
    {
        std::transform(pstl::execution::par_unseq, Input.Values.begin(), Input.Values.end(),
                       Output.Values.begin(), Function);
    }

private:
    UnaryFunction Function;
};

/*
// a custom function could look like this.
// this would have to be used inside the appropriate operation strategy functor
class WignerD
{
public:
    WignerD(Spin j, Spin muprime, Spin mu) {}

    std::complex<double> operator()(double theta, double phi) const 
    {
        // calculate value here
    }

private:
    Spin j;
    Spin muprime;
    Spin mu;
};*/

/*template <typename T>
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
};*/

using EdgeID = size_t;
using NodeID = size_t;
using DataID = size_t;

enum struct EdgeType
{
    DATA = 0,
    PARAMETER = 1,
    TEMPORARY = 2
};

struct FunctionGraphEdge
{
    EdgeID UniqueID;
    NodeID Source;
    NodeID Sink;
    EdgeType Type;
};

struct FunctionGraphNode
{
    NodeID UniqueID;
    std::unique_ptr<OperationStrategy> Operation;
};

// Represents a function, via a graph structure
// - nodes are operations
// - edges are data
// Note: edges can be attached to 1 or 2 nodes
// It implements the Function interface, and is one of the possible calculation
// backends.
// The nodes and edges show the hierarchy of the operations. The evaluate
// function calls the standard pipeline, which returns the appropriate result.
// This means no caching of intermediate results is performed, since it does
// not make sense for a function
// Note that it is assumed on evaluation that all inputs are already connected to the graph
template <typename OutputType>
class FunctionGraph : public Function<OutputType>
{
public:
    FunctionGraph() = default;
    virtual ~FunctionGraph() = default;

    template <typename Output, typename Input1, typename Input2, typename FunctionType = void(Output &, const Input1 &, const Input2 &)>
    EdgeID
    addBinaryNode(FunctionType Function, EdgeID InputID1, EdgeID InputID2)
    {
        EdgeID OutEdgeID = createIntermediateEdge<Output>({InputID1, InputID2});
        //std::cout << "binary size of output container: " << std::any_cast<const Output &>(getDataReference(OutEdgeID)).size() << std::endl;
        createNewNode(std::make_unique<BinaryOperationFunctor<Output, Input1, Input2, FunctionType>>(
            getDataReference<Output>(OutEdgeID),
            getDataReference<const Input1>(InputID1),
            getDataReference<const Input2>(InputID2),
            Function));
        return OutEdgeID;
    }

    template <typename Output, typename Input, typename FunctionType = void(Output &, const Input &)>
    EdgeID
    addUnaryNode(FunctionType Function, EdgeID InputID, bool CacheNode = false)
    {
        EdgeID OutEdgeID = createIntermediateEdge<Output>({InputID});
        //std::cout << "size of input container: " << std::any_cast<const Input &>(getDataReference(InputID)).size() << std::endl;
        //std::cout << "size of output container: " << std::any_cast<const Output &>(getDataReference(OutEdgeID)).size() << std::endl;
        auto a = getDataReference<Output>(OutEdgeID);
        createNewNode(std::make_unique<UnaryOperationFunctor<Output, Input, FunctionType>>(
            a,
            getDataReference<const Input>(InputID),
            Function));
        return OutEdgeID;
    }

    template <typename DataType>
    EdgeID createDataSource(DataType Data)
    {
        FunctionGraphEdge NewEdge;
        auto edgeid = getNewEdgeID();
        NewEdge.UniqueID = edgeid;
        NewEdge.Type = EdgeType::DATA;
        Edges.push_back(NewEdge);
        auto dataid = DataStorage.size();
        EdgeToDataMapping[edgeid] = dataid;
        DataStorage[dataid] = std::make_shared<DataType>(Data);
        return edgeid;
    }

    template <typename ParameterType>
    EdgeID createParameterEdge(ParameterType data)
    {
        FunctionGraphEdge NewEdge;
        auto edgeid = getNewEdgeID();
        NewEdge.UniqueID = edgeid;
        NewEdge.Type = EdgeType::PARAMETER;
        Edges.push_back(NewEdge);
        auto dataid = DataStorage.size();
        EdgeToDataMapping[edgeid] = dataid;
        DataStorage[dataid] = std::make_shared<ParameterType>(data);
        return edgeid;
    }

    virtual OutputType evaluate()
    {
        for (const auto &node : Nodes)
        {
            node.Operation->execute();
        }
        std::cout << "finished calc. returning data\n";
        auto b = *getDataReference<OutputType>(TopEdge);
        std::cout << "result size: " << b.Values.size() << std::endl;
        return b;
    }

    void updateParametersFrom(const ParameterList &list)
    {
        // the argument parameterlist does not have to contain the full set of parameters or?
        // because only a subset of parameters might be free. that means we would need a unique
        // way to identify a parameter. (maybe give it a unique id?)
    }

    ParameterList getParameters() const
    {
    }

    void fillDataContainers(const std::vector<std::vector<double>> &data)
    {
        //loop over the data containers, and fill them the data given here

        //(this procedure might also reshape them)
        // only call resizeDataContainers,
        // on branch parts where the data containers do not match in size
        presizeDataContainers();
    }

private:
    template <typename T>
    std::shared_ptr<T> getDataReference(EdgeID edgeid)
    {
        std::cout << "edgeid: " << edgeid << "\n";
        std::cout << "size mapping: " << EdgeToDataMapping.size() << "\n";
        auto a = EdgeToDataMapping.at(edgeid);
        std::cout << "dataid: " << a << "\n";
        std::cout << "datastorage size: " << DataStorage.size() << std::endl;
        return std::dynamic_pointer_cast<T>(DataStorage.at(a));
    }

    void presizeDataContainers(DataID id)
    {
        // basically we loop over the data containers
        // and if they are TEMPORARY and of dimension > 0
        // we resize them according to the input
        // if the input is not set, then we do the same for that data container....
        // (recursive??)
        //if DataStorage.at(id) ==)
    }

    template <typename T>
    EdgeID createIntermediateEdge(std::vector<EdgeID> InputEdgeIDs)
    {
        FunctionGraphEdge NewEdge;
        auto edgeid = getNewEdgeID();
        NewEdge.UniqueID = edgeid;
        NewEdge.Type = EdgeType::TEMPORARY;
        Edges.push_back(NewEdge);
        DataID dataid;
        bool FoundAvailableEdge(false);
        for (auto x : InputEdgeIDs)
        {
            /* auto result = std::find_if(Edges.begin(), Edges.end(), [&x](auto const &e) { return e.UniqueID == x; });
            if (result != Edges.end())
            {
                if (result->Type != EdgeType::TEMPORARY)
                {
                    // skip if not temporary data
                    continue;
                }
            }*/
            try
            {
                auto tempdata = getDataReference<T>(x);
            }
            catch (const std::bad_cast &e)
            {
                // if this is not the correct type of container, then just keep looking
                continue;
            }

            dataid = EdgeToDataMapping.at(x);
            FoundAvailableEdge = true;
            std::cout << "found good edge at " << dataid << "\n";
            break;
        }
        // if no suitable data container was found, create a new one
        if (!FoundAvailableEdge)
        {
            dataid = DataStorage.size();
            DataStorage[dataid] = std::make_shared<T>();
        }
        EdgeToDataMapping[edgeid] = dataid;

        TopEdge = edgeid;
        return edgeid;
    }

    EdgeID getNewEdgeID() const
    {
        return Edges.size();
    }

    NodeID createNewNode(std::unique_ptr<OperationStrategy> Op)
    {
        FunctionGraphNode NewNode;
        NewNode.UniqueID = getNewNodeID();
        NewNode.Operation = std::move(Op);
        Nodes.push_back(std::move(NewNode));
        return NewNode.UniqueID;
    }

    NodeID getNewNodeID() const
    {
        return Nodes.size();
    }

    std::vector<FunctionGraphNode> Nodes;
    std::vector<FunctionGraphEdge> Edges;
    // to reseat data elements, we need shared ptrs
    std::map<DataID, std::shared_ptr<Tensor>> DataStorage;
    std::map<EdgeID, DataID> EdgeToDataMapping;
    EdgeID TopEdge;
};

template <typename OutputType>
class FunctionGraphWrapper : public Function<OutputType, std::vector<DataPoint>>
{
public:
    OutputType evaluate(const std::vector<DataPoint> &DataPoints)
    {
        Graph.fillDataContainers(DataPoints);
        return Graph.evaluate();
    }

    void updateParametersFrom(const ParameterList &list)
    {
        Graph.updateParametersFrom(list);
    }

    ParameterList getParameters() const
    {
        return Graph.getParameters();
    }

private:
    FunctionGraph<OutputType> Graph;
};

// IDEA: I think its best if I leave the FunctionGraph simple, and just like a function, that can be evaluated
// If I want to do fitting I need an estimator, which would wrap/decorate this FunctionGraph and do all of the
// create pipelines, parameter changed -> calculate mask, determine which pipeline to run etc...
// So at this point I have the information about the parameters (if fixed or not). this estimator (decorator) gets the additional function
// setParameterFitSettings() which sets a parameter fixed or not, then the full parameter set is just reduced to the non fixed ones
// This means that this new estimator would also limit itself to our own functiongraph version. but i don't think that is bad....
//
// another thing is how is the whole graph going to be executed.
// for example: we can have a set of pipelines (a branch pipeline, which would be just a vector of connected operations)
// depending on the parameters which changed. then at each fit iteration, we just execute the pipeline that corresponds
// to that change and run that pipeline. the pipeline itself does not take much storage in memory, so having multiple of the does not matter
// then res should be just the vector of the results that we want to return from the interface
// IMPORTANT: the mask mentioned above is crucial. So we have a set of branch or partial pipelines, which make up the full graph.
// Every parameter maps to one of these branch pipelines, this is surjective. With a mask we can determine if a certain partial pipeline
// should be executed in the next evaluation in an efficient way.
template <typename OutputType>
class FunctionGraphEvaluator : public Function<OutputType>
{
public:
    // this will take a functiongraph and data, and connect everything in a fixed way
    FunctionGraphEvaluator(FunctionGraph<OutputType> g) : Graph(g)
    {
    }

    void performOptimizations()
    {
        // perform graph optimizations, such as
        // - dynamic caching of intermediate edges
        // - presize all intermediate edges accordingly
        // - reseat data containers if possible to reduce memory usage (without speed loss)
        // calculate the mapping between parameter and partial/branch pipelines
    }

    void createPipelines()
    {
        // few things that are important:
        // 1. we want to shadow graph branches (up to their leaves), which do not have any
        // non-fixed parameter leaves. Just add a cached node on top of that branch since this
        // value is always constant during the fit. This has to be optional though, since it might
        // take to much memory.
        // 2. when parameters become fixed or non-fixed, different parts of the graph
        // have to be precalculated and shadowed
        // 3. the nodes have to be in a hierarchy, so that you know which node has to be calculated
        // before another etc. This is easy though, since you have all of the connections (edges)
        // 4. so in the initialization phase, we create a map from changed parameter mask to pipeline
        // Then we simply have to execute that pipeline when we got a new parameter set
        // 5. some data might be used in multiple nodes, in that case it would make sense to cache
        // that cache that data automatically
        // 6. data which is cached represents the endpoint of one pipeline, and the entrypoint for
        // new branch pipelines.
        // 7. a branch pipeline calculates only a certain part of the full graph
        // (but all of the intermediate datas are temporary)
    }

    //void attachFunctionGraphToEdge(edgeid id, functiongraph g, datavectors data)
    //{
    // get nodes, edges from that graph and incorporate into this graph
    //}

    OutputType evaluate()
    {
        //Here we do not call the evaluate of the functiongraph
        // then just process the pipelines we created before (depending on which parameters changed)
        // after the pipeline call, we switch back to the default pipeline (assumes nothing changed? so just returns result)
    }

    void updateParametersFrom(const ParameterList &list)
    {
        // update all parameters
        // at the same time determine which parameters have changed
        // then select a new current pipeline, depending on the list of changed parameters
    }

    ParameterList getParameters() const
    {
        // just call the getParameters function from the functiongraph
    }

private:
    FunctionGraph<OutputType> Graph;
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
    size_t loops(1);

    FunctionGraph<Vector<double>> g;

    auto ida = g.createDataSource(Vector<double>(a));
    auto idb = g.createDataSource(Vector<double>(b));

    auto mycos_wrapper = [](double x) { return std::cos(x); };
    auto myabs_wrapper = [](double x) { return std::abs(x); };
    auto mysqrt_wrapper = [](double x) { return std::sqrt(x); };

    auto blub = ElementWiseUnaryOperation<decltype(mycos_wrapper)>(mycos_wrapper);
    auto asdf = ElementWiseBinaryOperation<decltype(std::multiplies<>())>(std::multiplies<>());

    auto tempres1 = g.addUnaryNode<Vector<double>, Vector<double>>(blub, ida);
    auto myparam = g.createParameterEdge(Scalar<double>(3.0));
    auto tempres11 = g.addBinaryNode<Vector<double>, Vector<double>, Scalar<double>>(asdf, tempres1, myparam);
    auto tempres2 = g.addBinaryNode<Vector<double>, Vector<double>, Vector<double>>(asdf, tempres11, idb);
    auto tempres3 = g.addUnaryNode<Vector<double>, Vector<double>>(ElementWiseUnaryOperation<decltype(myabs_wrapper)>(myabs_wrapper), tempres2);
    auto res = g.addUnaryNode<Vector<double>, Vector<double>>(ElementWiseUnaryOperation<decltype(mysqrt_wrapper)>(mysqrt_wrapper), tempres3);

    //g.fillDataContainers({a, b});

    for (size_t i = 0; i < loops; ++i)
    {
        std::chrono::steady_clock::time_point StartTime =
            std::chrono::steady_clock::now();

        auto result = g.evaluate().Values;

        std::chrono::steady_clock::time_point EndTime =
            std::chrono::steady_clock::now();

        sec += (EndTime - StartTime);

        for (size_t i = 0; i < vecsize; ++i)
        {
            assert(std::abs(result[i] - std::sqrt(std::abs(3 * std::cos(a[i]) * b[i]))) < 1e-10);
        }
    }
    sec /= loops;
    std::cout << sec.count() << " seconds " << std::endl;
}