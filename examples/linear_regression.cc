import rahml;
import <iostream>;
import <vector>;
import <cassert>;

using rahml::Value;

int main() {
    // we are trying to get our model to predict y = 2x + 1
    const std::vector<double> xs{-2.0, -1.0, 0.0, 1.0, 2.0};
    double weight = 0.0;
    double bias = 0.0; 
    const double learning_rate = 0.1;

    for (int step = 0; step < 100; ++step) {
        Value w{weight}; // weight
        Value b{bias}; // bias
        Value total_loss{0.0};
        for (double x_num : xs) {
            Value x{x_num};
            Value target{2.0 * x_num + 1.0};
            Value prediction = w * x + b;
            Value error = prediction - target;
            total_loss = total_loss + error * error; // add it on to running sum
        }
        // loss is the mean squared error
        Value loss = total_loss / Value{static_cast<double>(xs.size())};
        loss.backward(); // run backprop
        if (step % 10 == 0) {
            std::cout << "step = " << step << ", loss = " << loss.data() <<
            ", w = " << w.data() << ", b = " << b.data() << "\n";
        }
        weight -= learning_rate * w.grad();
        bias -= learning_rate * b.grad();
    }
    std::cout << "learned w = " << weight << '\n';
    std::cout << "learned b = " << bias << '\n';

    assert(weight > 1.999 && weight < 2.001);
    assert(bias > 0.999 && bias < 1.001);
}