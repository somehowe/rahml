import <cmath>;
import <random>;
import <vector>;
import rahml;

// 1. A client adding a new math operation (sin) to the computation graph:
// Internally, Node is a base class with a pure virtual backward_step() method.
// unary_op is a template function that creates a derived UnaryOpNode<Fn> holding
// the client's derivative function so backward() knows how to differentiate it.
rahml::Value sin_val(const rahml::Value& x) {
    double val = std::sin(x.data());
    return rahml::unary_op(x, val, "sin", [](double out_grad, double x_data) {
        // Chain rule for sin(x): incoming gradient * cos(x)
        return out_grad * std::cos(x_data);
    });
}

// 2. A client defining their own custom layer using the Module base class:
class ScaledSineSkip : public rahml::Module {
    rahml::Value scale_ = rahml::Value::parameter(0.1);

public:
    std::vector<rahml::Value> forward(
        const std::vector<rahml::Value>& inputs) override {
        std::vector<rahml::Value> outputs;
        outputs.reserve(inputs.size());
        for (const auto& x : inputs) {
            // Mixes built-in operators (+, *) with the custom sin_val operation.
            // x is used along two paths, and scale_ is shared across the vector.
            outputs.push_back(x + scale_ * sin_val(x));
        }
        return outputs;
    }

    std::vector<rahml::Value> parameters() const override {
        return {scale_}; // Copying this handle still refers to the same parameter.
    }
};

int main() {
    std::mt19937 rng{42};
    std::vector<rahml::Example> xor_data = {
        {{0.0, 0.0}, {-1.0}},
        {{0.0, 1.0}, { 1.0}},
        {{1.0, 0.0}, { 1.0}},
        {{1.0, 1.0}, {-1.0}},
    };

    rahml::Sequential model;
    auto& block = model.emplace<rahml::Sequential>();
    block.emplace<rahml::Linear>(2, 3, rng);
    block.emplace<rahml::Tanh>();
    model.emplace<ScaledSineSkip>();
    model.emplace<rahml::Linear>(3, 1, rng);

    rahml::MeanSquaredError loss;
    rahml::SGD optimizer{model.parameters(), 0.05};
    rahml::Trainer trainer{model, loss, optimizer};
    trainer.fit(xor_data, 1000, 4, rng);
}
